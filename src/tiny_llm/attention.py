import math
import mlx.core as mx
from .basics import softmax, linear


def scaled_dot_product_attention_simple(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | None = None,
) -> mx.array:
    # 1. Accquire Dimension(head_dim)
    # query shape: (B, num_heads, L, D)意思是最后一个维度是D
    D = query.shape[-1]
    # 2. Calculate (1/sqrt(D))
    if scale is None:
        scale = 1.0 / (math.sqrt(D))
    # 3. Calculate Scores = Q @ K^T * scale
    # key的形状: (B, num_heads, S, D)，我们需要转置最后两个维度，目的是为了做矩阵乘法
    scores = (query @ key.swapaxes(-1, -2)) * scale
    # 4. Apply Mask (if provided)
    if mask is not None:
        scores = scores + mask
    # 5. Apply Softmax to get Attention Weights
    attn_weights = mx.softmax(scores, axis=-1)  # shape: (B, num_heads, L, S)
    # 6. Compute Output = Attention Weights @ V
    output = attn_weights @ value  # shape: (B, num_heads, L
    return output


class SimpleMultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        # 计算每个头的维度 D = hidden_size / num_heads
        self.head_dim = hidden_size // num_heads
        #保存线性变换权重
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo

    def __call__(
        self,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        mask: mx.array | None = None,
    ) -> mx.array:
        # 输入维度: (B, L, E)
        # 1. 线性投影
        # 此时形状： (B, L, hidden_size) -> (B, L, num_heads * head_dim)
        # 这里的query,key,value可能来自于上一层，通常它们是同一个输入 x（self-attention）
        q = linear(query, self.wq)  # shape: (B, L, hidden_size)
        k = linear(key, self.wk)
        v = linear(value, self.wv)
        # 2. 拆分头
        # Reshape and transpose to get (B, num_heads, L, head_dim)
        B, L, _ = q.shape
        q = q.reshape(B, L, self.num_heads, self.head_dim)
        k = k.reshape(B, L, self.num_heads, self.head_dim)
        v = v.reshape(B, L, self.num_heads, self.head_dim)
        # Transpose to (B, num_heads, L, head_dim)
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        # 3. 计算注意力
        attn_output = scaled_dot_product_attention_simple(q, k, v, mask=mask)
        # 4. 合并头
        # Transpose back to (B, L, num_heads, head_dim)
        attn_output = attn_output.transpose(0, 2, 1, 3)
        # Reshape to (B, L, hidden_size)
        attn_output = attn_output.reshape(B, L, self.hidden_size)
        # 5. 最后的线性变换
        output = linear(attn_output, self.wo)  # shape: (B, L, hidden_size)

        return output


def causal_mask(L: int, S: int, dtype: mx.Dtype) -> mx.array:
    # 1. 生成Key/Value的位置索引: （1，S）
    # 含义是代表每一条信息的时间戳位置
    S_indices = mx.arange(S)[None, :]  # shape: (1, S)

    # 2. 生成Query的位置索引: (L, 1) 并进行“右对齐”修正
    # 含义是代表当前查询的时间戳位置
    # 核心逻辑(S - L): 模拟KV Cache机制下，Query位置相对于Key/Value位置的偏移
    L_indices = mx.arange(L)[:, None] + (S - L)

    # 3. 计算掩码矩阵: (L, S)
    # 公式: mask[i, j] = 1 if j <= i + (S - L) else -inf
    # 这里使用广播机制来实现比较操作
    mask = mx.where(
        S_indices > L_indices, -mx.inf, 0.0
    )

    # 4. 转换为指定的数据类型
    return mask.astype(dtype)


def scaled_dot_product_attention_grouped(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | str | None = None,
) -> mx.array:
    #query (B, H_q, L, D)
    #key (B, H_k, S, D)
    *batch, H_q, L, D = query.shape
    H_kv = key.shape[-3]
    S = key.shape[-2]
    n_repeats = H_q // H_kv

    # Reshape Q to group heads
    # new shape: (B, H_kv, n_repeats, L, D)
    query = query.reshape(*batch, H_kv, n_repeats, L, D)
    
    # Reshape K and V to add head repeat dimension
    # new shape: (B, H_kv, 1, S, D)
    key = key.reshape(*batch, H_kv, 1, S, D)
    value = value.reshape(*batch, H_kv, 1, S, D)
    #Calculate scale
    if scale is None:
        scale = 1.0 / (math.sqrt(D))

    #Calculate Scaled Dot-Product Attention
    scores = (query @ key.swapaxes(-1, -2)) * scale
    # 情况 A: 用户传入了现成的 Mask 矩阵 (mx.array)
    if isinstance(mask, mx.array):
        mask = mask.reshape(*batch, H_kv, n_repeats, L, S)
        scores = scores + mask

    # 情况 B: 用户指定需要因果掩码 (str="causal")
    # TinyLLM 课程中，有时候会传入字符串 "causal" 来指示函数内部自动生成掩码
    elif mask == "causal":
        # 调用刚才写好的函数
        # 注意：这里生成的 mask 形状是 (L, S)，它会自动广播加到 (B, H, n, L, S) 上
        c_mask = causal_mask(L, S, scores.dtype)
        scores = scores + c_mask
    probs = mx.softmax(scores, axis=-1)
    output = probs @ value  # shape: (B, H_kv, n_repeats, L, D)
    # Reshape back to (B, H_q, L, D)
    output = output.reshape(*batch, H_q, L, D)  # shape: (B, H_q, L, D)

    return output



def flash_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | None = None,
) -> mx.array:
    pass
