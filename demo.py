"""Run the secret-shared two-layer network and compare it with plaintext."""

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

    print("Two-party secret-shared two-layer network")
    print("h = (W1 @ x + b1) ** 2")
    print("y = W2 @ h + b2")
    print()
    print("By hand: z = [1*1 + 0*2, 1*1 + (-1)*2 + 1] = [1, 0]")
    print("         h = [1, 0]")
    print("         y = [2*1 + 3*0 + 4] = [6]")
    print()

    plaintext = forward(
        network["x"], network["w1"], network["b1"], network["w2"], network["b2"]
    )
    secure = secure_forward(
        shared["x"], shared["w1"], shared["b1"], shared["w2"], shared["b2"], dealer
    )
    opened = reveal(secure)

    show("input x", network["x"])
    show("party 0 input share", [to_signed(v) for v in shared["x"].s0])
    show("party 1 input share", [to_signed(v) for v in shared["x"].s1])
    print("One share is uniform in the field. It does not reveal x, the weights, or the output.")
    print()
    show("plaintext output", plaintext)
    show("party 0 output share", [to_signed(v) for v in secure.s0])
    show("client opens both output shares", opened)
    print(f"multiplications: {dealer.multiplications}")

    if [int(v) for v in opened] != [int(v) for v in plaintext]:
        print("secure forward does not match plaintext", file=sys.stderr)
        sys.exit(1)
    print("Secure forward matches plaintext.")


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
    print("Random small network (input 3, hidden 4, output 2)")
    show("plaintext output", plaintext)
    show("secret-shared output", opened)
    if [int(v) for v in opened] != [int(v) for v in plaintext]:
        print("random network comparison failed", file=sys.stderr)
        sys.exit(1)
    print("Random network matches.")


if __name__ == "__main__":
    run_example()
    run_random()
