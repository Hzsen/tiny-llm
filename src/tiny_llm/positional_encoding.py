import mlx.core as mx


class RoPE:
    def __init__(
        self,
        dims: int,
        seq_len: int,
        base: int = 10000,
        traditional: bool = False,
    ):
        self.dim = dims
        
        # 1. 计算 theta 频率 (只计算 D/2 个)
        # range: [0, 1, ..., dim/2 - 1]
        # 公式: theta = base ^ (-2 * i / dim)
        indices = mx.arange(0, dims // 2, dtype=mx.float32)
        theta = base ** (-2 * indices / dims)
        
        # 2. 生成位置索引 (Positions)
        # range: [0, 1, ..., max_seq_len - 1]
        positions = mx.arange(0, seq_len, dtype=mx.float32)
        
        # 3. 计算所有位置的角度 (Angles) = Position * Theta
        # 利用广播/外积: (L, 1) * (1, D/2) -> (L, D/2)
        angles = positions[:, None] * theta[None, :]
        
        # 4. 缓存 cos 和 sin
        # 形状都是 (MAX_SEQ_LEN, D // 2)
        self.cos_freqs = mx.cos(angles)
        self.sin_freqs = mx.sin(angles)

    def __call__(
        self, x: mx.array, offset: list[slice] | slice | None = None
    ) -> mx.array:
        # 输入 x 形状: (Batch, L, Heads, Dim)
        # 在 Qwen2 的实现中，我们不需要 reshape 成 (..., 2)
        # 而是直接把 Dim 维度切成两半
        
        N, L, H, D = x.shape
        half_dim = D // 2
        
        # 1. 获取对应的 frequencies
        if offset is None:
            cos = self.cos_freqs[:L]
            sin = self.sin_freqs[:L]
        else:
            cos = self.cos_freqs[offset]
            sin = self.sin_freqs[offset]
            
        # 2. 调整 frequencies 形状以支持广播
        # cos/sin 原本形状: (L, D//2)
        # 目标广播形状: (1, L, 1, D//2)
        cos = cos.reshape(1, L, 1, half_dim)
        sin = sin.reshape(1, L, 1, half_dim)
        
        # 3. 将输入 x 切分为前后两半
        # x1: 前半部分 (..., :half_dim)
        # x2: 后半部分 (..., half_dim:)
        x1 = x[..., :half_dim]
        x2 = x[..., half_dim:]
        
        # 4. 应用非传统 RoPE 旋转公式
        # 根据题目公式:
        # output[0] (前半部分对应位) = x1 * cos + x2 * (-sin)
        # output[half] (后半部分对应位) = x1 * sin + x2 * cos
        
        out1 = x1 * cos - x2 * sin
        out2 = x1 * sin + x2 * cos
        
        # 5. 拼接回原始形状
        # 在最后一个维度 (Dim) 上拼接
        return mx.concatenate([out1, out2], axis=-1)
