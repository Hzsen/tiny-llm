import mlx.core as mx
import math


def softmax(x: mx.array, axis: int) -> mx.array:
    # TODO: manual implementation
    return mx.softmax(x, axis=axis)


def linear(
    x: mx.array,
    w: mx.array,
    bias: mx.array | None = None,
) -> mx.array:
    # 1. 矩阵乘法
    # x: (..., input_dim)
    # w: (output_dim, input_dim) -> 需要转置为 (input_dim, output_dim)
    output = x @ w.T
    
    # 2. 加上偏置 (如果有)
    if bias is not None:
        output = output + bias
        
    # 3. 【重要】必须返回结果！
    return output


def silu(x: mx.array) -> mx.array:
    # SiLU (Sigmoid Linear Unit) 激活函数
    # 公式: x * sigmoid(x) = x / (1 + e^-x)
    return x * mx.sigmoid(x)
