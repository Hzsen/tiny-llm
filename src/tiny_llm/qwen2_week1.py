import mlx.core as mx
from .basics import linear, silu
from .attention import scaled_dot_product_attention_grouped
from .layer_norm import RMSNorm
from .positional_encoding import RoPE
from typing import Any
from .embedding import Embedding
from .quantize import dequantize_linear


class Qwen2MultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
        bq: mx.array,
        bk: mx.array,
        bv: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads

        # 保存线性变换权重和偏置
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo
        self.bq = bq
        self.bk = bk
        self.bv = bv

        #initialize RoPE
        self.rope = RoPE(
            dims=self.head_dim,
            seq_len=max_seq_len,
            base=theta,
            traditional=False,
        )

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        # 1.线性投影 Q,K,V（带偏置）
        B, L, _ = x.shape
        q = linear(x, self.wq, self.bq)  # shape: (B, L, hidden_size)
        k = linear(x, self.wk, self.bk)
        v = linear(x, self.wv, self.bv)

        # 2.Reshape and Transpose for Multi-Head Attention
        q = q.reshape(B, L, self.num_heads, self.head_dim)
        k = k.reshape(B, L, self.num_kv_heads, self.head_dim)
        v = v.reshape(B, L, self.num_kv_heads, self.head_dim)

        # 3. Apply RoPE to Q and K
        q = self.rope(q, offset=slice(0, L))
        k = self.rope(k, offset=slice(0, L))

        # 4. Set up Attention using grouped attention function
        q = q.transpose(0, 2, 1, 3)  # shape: (B, num_heads, L, head_dim)
        k = k.transpose(0, 2, 1, 3)  # shape: (B, num_kv_heads, L, head_dim)
        v = v.transpose(0, 2, 1, 3)  # shape: (B, num_kv_heads, L, head_dim)


        # 5. Compute Attention Output and compel float32 output
        output = scaled_dot_product_attention_grouped(
            q.astype(mx.float32),
            k.astype(mx.float32),
            v.astype(mx.float32),
            mask=mask,
        )

        # 6. Rotate and reshape back
        output = output.astype(x.dtype)
        output = output.transpose(0, 2, 1, 3)
        output = output.reshape(B, L, self.hidden_size)

        # 7. Final linear projection
        output = linear(output, self.wo)

        return output


class Qwen2MLP:
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        w_gate: mx.array,
        w_up: mx.array,
        w_down: mx.array,
    ):
        self.w_gate = w_gate
        self.w_up = w_up
        self.w_down = w_down

    def __call__(self, x: mx.array) -> mx.array:
        # SwiGLU 实现:
        # output = (SiLU(Gate(x)) * Up(x)) -> Down(...)
        
        gate = linear(x, self.w_gate)
        up = linear(x, self.w_up)
        
        # SiLU 激活
        act = silu(gate)
        
        # Element-wise 乘法
        h = act * up
        
        # Down projection
        output = linear(h, self.w_down)
        return output


class Qwen2TransformerBlock:
    def __init__(
        self,
        num_attention_heads: int,
        num_kv_heads: int,
        hidden_size: int,
        intermediate_size: int,
        rms_norm_eps: float,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
        bq: mx.array,
        bk: mx.array,
        bv: mx.array,
        w_gate: mx.array,
        w_up: mx.array,
        w_down: mx.array,
        w_input_layernorm: mx.array,
        w_post_attention_layernorm: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
    ):
        # 初始化 Attention
        self.self_attn = Qwen2MultiHeadAttention(
            hidden_size=hidden_size,
            num_heads=num_attention_heads,
            num_kv_heads=num_kv_heads,
            wq=wq, wk=wk, wv=wv, wo=wo,
            bq=bq, bk=bk, bv=bv,
            max_seq_len=max_seq_len,
            theta=theta
        )
        # 初始化 MLP
        self.mlp = Qwen2MLP(
            dim=hidden_size,
            hidden_dim=intermediate_size,
            w_gate=w_gate,
            w_up=w_up,
            w_down=w_down
        )
        # 初始化 RMSNorm
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.input_layernorm.weight = w_input_layernorm
        
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm.weight = w_post_attention_layernorm

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        # 1. Self Attention (Pre-Norm)
        # Residual connection: x + Attention(Norm(x))
        h = x + self.self_attn(self.input_layernorm(x), mask=mask)
        
        # 2. MLP (Pre-Norm)
        # Residual connection: h + MLP(Norm(h))
        output = h + self.mlp(self.post_attention_layernorm(h))
        
        return output


class Qwen2ModelWeek1:
    def __init__(self, mlx_model: Any):
        self.model = mlx_model

    def __call__(
        self,
        inputs: mx.array,
    ) -> mx.array:
        pass
