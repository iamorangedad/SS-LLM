"""运行两层秘密分享网络，并与明文前向对照。"""

import sys

import numpy as np

from ssnn import Dealer, example, forward, reveal, secure_forward, to_signed


def show(title: str, values) -> None:
    rendered = ", ".join(str(int(v)) for v in np.asarray(values, dtype=object).reshape(-1))
    print(f"{title}: [{rendered}]")


def run_example() -> None:
    network = example()
    dealer = Dealer(seed=2026)
    shared = {
        name: dealer.share(network[name]) for name in ("x", "w1", "b1", "w2", "b2")
    }

    print("两方秘密分享上的两层网络")
    print("h = (W1 @ x + b1) ** 2")
    print("y = W2 @ h + b2")
    print()
    print("手算：z = [1*1 + 0*2, 1*1 + (-1)*2 + 1] = [1, 0]")
    print("      h = [1, 0]")
    print("      y = [2*1 + 3*0 + 4] = [6]")
    print()

    plaintext = forward(
        network["x"], network["w1"], network["b1"], network["w2"], network["b2"]
    )
    secure = secure_forward(
        shared["x"], shared["w1"], shared["b1"], shared["w2"], shared["b2"], dealer
    )
    opened = reveal(secure)

    show("输入 x", network["x"])
    show("参与方 0 持有的输入份额", [to_signed(v) for v in shared["x"].s0])
    show("参与方 1 持有的输入份额", [to_signed(v) for v in shared["x"].s1])
    print("单独一份份额是域上的均匀随机数，合并之前还原不出 x、权重或输出。")
    print()
    show("明文输出", plaintext)
    show("参与方 0 持有的输出份额", [to_signed(v) for v in secure.s0])
    show("客户端合并两份输出份额", opened)
    print(f"乘法次数: {dealer.multiplications}")

    if [int(v) for v in opened] != [int(v) for v in plaintext]:
        print("安全前向与明文不一致", file=sys.stderr)
        sys.exit(1)
    print("安全前向与明文一致。")


def run_random() -> None:
    rng = np.random.default_rng(7)
    features = rng.integers(-4, 5, size=3).tolist()
    weight1 = rng.integers(-4, 5, size=(4, 3)).tolist()
    bias1 = rng.integers(-4, 5, size=4).tolist()
    weight2 = rng.integers(-4, 5, size=(2, 4)).tolist()
    bias2 = rng.integers(-4, 5, size=2).tolist()
    dealer = Dealer(seed=11)
    opened = reveal(
        secure_forward(
            dealer.share(features),
            dealer.share(weight1),
            dealer.share(bias1),
            dealer.share(weight2),
            dealer.share(bias2),
            dealer,
        )
    )
    plaintext = forward(features, weight1, bias1, weight2, bias2)
    print()
    print("随机小网络（输入 3，隐藏 4，输出 2）")
    show("明文输出", plaintext)
    show("秘密分享输出", opened)
    if [int(v) for v in opened] != [int(v) for v in plaintext]:
        print("随机网络对照失败", file=sys.stderr)
        sys.exit(1)
    print("随机网络对照一致。")


if __name__ == "__main__":
    run_example()
    run_random()
