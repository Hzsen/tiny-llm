import mlx.core as mx


class RMSNorm:
    def __init__(self, dim: int, weight: mx.array, eps: float = 1e-5):
        '''
        initerialize RMSNorm layer.
        
        :param self: Description
        :param dim: Description
        :type dim: int
        :param weight: Description
        :type weight: mx.array
        :param eps: Description
        :type eps: float
        '''
        self.dim = dim
        self.eps = eps
        self.weight = weight
        

    def __call__(self, x: mx.array) -> mx.array:
        # 1. 类型转换
        original_dtype = x.dtype
        x = x.astype(mx.float32)
        # 2. Calculate Mean Square
        mean_square = mx.mean(x**2, axis=-1, keepdims=True)
        # 3. Calculate Reciprocal RMS
        r_rms = mx.rsqrt(mean_square + self.eps)
        # 4. Normalize
        normed = x * r_rms * self.weight
        # 5. 类型还原并返回
        return normed.astype(x.dtype)
