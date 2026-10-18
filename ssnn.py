"""两方加法秘密分享上的两层神经网络。

这是一个最小、可核对的推理 demo，不是生产级多方计算库。

安全模型
    半诚实、两方不串通。每一方只持有自己的份额，并看到 Beaver 乘法打开的掩码。
    掩码由可信初始化方预先生成，初始化方不参与推理。
    份额在素数域 2^61-1 上均匀抽取，单份与秘密信息论独立。
    输入、权重、偏置和输出在协议结束前都不合并。输出只由客户端合并。

网络（整数算术）
    h = (W1 @ x + b1) ** 2
    y = W2 @ h + b2

平方是非线性，而且只需要乘法。ReLU 需要比较协议，超出这个最小实现。
域足够大，demo 使用的小整数在计算过程中不会绕模，因此结果与普通整数前向一致。
"""

from dataclasses import dataclass

import numpy as np

PRIME = 2**61 - 1


@dataclass
class Shared:
    """一个张量的两份份额。s0 归参与方 0，s1 归参与方 1。"""

    s0: np.ndarray
    s1: np.ndarray

    @property
    def shape(self):
        if self.s0.shape != self.s1.shape:
            raise ValueError("双方份额形状不一致")
        return self.s0.shape


class Dealer:
    """可信初始化方：只负责拆分秘密和发放乘法三元组。"""

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.multiplications = 0

    def share(self, secret) -> Shared:
        secret = field(secret)
        mask = self._uniform(secret.shape)
        return Shared(mask, (secret - mask) % PRIME)

    def _uniform(self, shape) -> np.ndarray:
        size = int(np.prod(shape, dtype=int))
        drawn = [int(v) for v in self.rng.integers(0, PRIME, size=size)]
        return np.array(drawn, dtype=object).reshape(shape)


def field(values) -> np.ndarray:
    """把整数张量规范化到域元素。负数按 Python 取模，变成对应的域元素。"""

    array = np.asarray(values, dtype=object)
    encoded = [int(v) % PRIME for v in array.reshape(-1)]
    return np.array(encoded, dtype=object).reshape(array.shape)


def to_signed(value) -> int:
    reduced = int(value) % PRIME
    if reduced > PRIME // 2:
        return reduced - PRIME
    return reduced


def reveal(shared: Shared) -> np.ndarray:
    """客户端合并两份份额。协议运行期间参与方不调用它。"""

    total = (shared.s0 + shared.s1) % PRIME
    signed = [to_signed(v) for v in np.asarray(total, dtype=object).reshape(-1)]
    return np.array(signed, dtype=object).reshape(shared.shape)


def _party_mask(share: np.ndarray, triple_share: np.ndarray) -> np.ndarray:
    """参与方本地计算将要打开的掩码份额。"""

    return (share - triple_share) % PRIME


def _party_product(opened_e, opened_f, a, b, c, add_cross: bool) -> np.ndarray:
    """参与方用公开掩码和自己的三元组份额完成本地乘法。"""

    result = (c + opened_e * b + opened_f * a) % PRIME
    if add_cross:
        result = (result + opened_e * opened_f) % PRIME
    return result


def multiply(left: Shared, right: Shared, dealer: Dealer) -> Shared:
    """Beaver 乘法。双方唯一互相发送的是掩码份额，打开后得到 e、f。"""

    if left.shape != right.shape:
        raise ValueError("相乘的两个份额形状必须相同")

    shape = left.shape
    raw_a = dealer._uniform(shape)
    raw_b = dealer._uniform(shape)
    raw_c = (raw_a * raw_b) % PRIME
    triple_a = dealer.share(raw_a)
    triple_b = dealer.share(raw_b)
    triple_c = dealer.share(raw_c)
    dealer.multiplications += int(np.prod(shape, dtype=int))

    opened_e = (
        _party_mask(left.s0, triple_a.s0) + _party_mask(left.s1, triple_a.s1)
    ) % PRIME
    opened_f = (
        _party_mask(right.s0, triple_b.s0) + _party_mask(right.s1, triple_b.s1)
    ) % PRIME
    product0 = _party_product(
        opened_e, opened_f, triple_a.s0, triple_b.s0, triple_c.s0, add_cross=True
    )
    product1 = _party_product(
        opened_e, opened_f, triple_a.s1, triple_b.s1, triple_c.s1, add_cross=False
    )
    return Shared(product0, product1)


def add(left: Shared, right: Shared) -> Shared:
    if left.shape != right.shape:
        raise ValueError("相加的两个份额形状必须相同")
    return Shared((left.s0 + right.s0) % PRIME, (left.s1 + right.s1) % PRIME)


def square(value: Shared, dealer: Dealer) -> Shared:
    return multiply(value, value, dealer)


def matvec(weight: Shared, vector: Shared, dealer: Dealer) -> Shared:
    """秘密分享下的矩阵乘向量。权重和向量都是份额。"""

    if weight.s0.ndim != 2 or vector.s0.ndim != 1:
        raise ValueError("matvec 需要 (输出, 输入) 矩阵和一维向量")
    rows, cols = weight.shape
    if vector.shape != (cols,):
        raise ValueError("矩阵列数必须等于向量长度")

    expanded = Shared(
        np.broadcast_to(vector.s0, (rows, cols)),
        np.broadcast_to(vector.s1, (rows, cols)),
    )
    products = multiply(weight, expanded, dealer)
    summed0 = _sum_rows(products.s0)
    summed1 = _sum_rows(products.s1)
    return Shared(summed0, summed1)


def _sum_rows(matrix: np.ndarray) -> np.ndarray:
    rows = []
    for row in np.asarray(matrix, dtype=object):
        total = 0
        for entry in row:
            total += int(entry)
        rows.append(total % PRIME)
    return np.array(rows, dtype=object)


def secure_forward(
    features: Shared,
    weight1: Shared,
    bias1: Shared,
    weight2: Shared,
    bias2: Shared,
    dealer: Dealer,
) -> Shared:
    """在份额上计算两层网络，不合并任何中间结果。"""

    hidden_input = add(matvec(weight1, features, dealer), bias1)
    hidden = square(hidden_input, dealer)
    return add(matvec(weight2, hidden, dealer), bias2)


def forward(features, weight1, bias1, weight2, bias2) -> np.ndarray:
    """同一网络的明文整数前向，用来对照安全计算的结果。"""

    features = [int(v) for v in np.asarray(features, dtype=object).reshape(-1)]
    weight1 = np.asarray(weight1, dtype=object)
    bias1 = [int(v) for v in np.asarray(bias1, dtype=object).reshape(-1)]
    weight2 = np.asarray(weight2, dtype=object)
    bias2 = [int(v) for v in np.asarray(bias2, dtype=object).reshape(-1)]

    if weight1.ndim != 2 or weight1.shape[1] != len(features):
        raise ValueError("第一层权重形状与输入不符")
    if len(bias1) != weight1.shape[0]:
        raise ValueError("第一层偏置长度与隐藏层不符")

    hidden = []
    for row, bias in zip(weight1, bias1):
        activation = bias
        for weight, feature in zip(row, features):
            activation += int(weight) * feature
        hidden.append(activation * activation)

    if weight2.ndim != 2 or weight2.shape[1] != len(hidden):
        raise ValueError("第二层权重形状与隐藏层不符")
    if len(bias2) != weight2.shape[0]:
        raise ValueError("第二层偏置长度与输出不符")

    output = []
    for row, bias in zip(weight2, bias2):
        value = bias
        for weight, activation in zip(row, hidden):
            value += int(weight) * activation
        output.append(value)
    return np.array(output, dtype=object)


def example():
    """手算例子：y = [6]。推导见 demo.py。"""

    return {
        "x": [1, 2],
        "w1": [[1, 0], [1, -1]],
        "b1": [0, 1],
        "w2": [[2, 3]],
        "b2": [4],
    }
