import mlx.core as mx


class Embedding:
    def __init__(self, vocab_size: int, embedding_dim: int, weight: mx.array):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.weight = weight

    def __call__(self, x: mx.array) -> mx.array:
        return self.weight[x]

    def as_linear(self, x: mx.array) -> mx.array:
        # 通过矩阵乘法实现与 Embedding 相同的功能
        # x 的形状: (..., embedding_dim)
        # weight 的形状: (vocab_size, embedding_dim)
        # 我们需要转置 weight 以进行矩阵乘法
        logits = x @ self.weight.T  # 形状: (..., vocab_size)
        return logits
