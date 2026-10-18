"""核对秘密分享、Beaver 乘法和两层前向。"""

import numpy as np

from ssnn import Dealer, PRIME, example, forward, multiply, reveal, secure_forward


def values(array):
    return [int(v) for v in np.asarray(array, dtype=object).reshape(-1)]


def test_share_roundtrip_and_hiding():
    secret = [1, -2, 3]
    seen = set()
    for seed in range(30):
        shared = Dealer(seed).share(secret)
        assert values(reveal(shared)) == secret
        party = tuple(int(v) for v in shared.s0)
        assert party != tuple(int(v) % PRIME for v in secret)
        seen.add(party)
    assert len(seen) == 30


def test_beaver_matches_product():
    dealer = Dealer(3)
    left = dealer.share([3, -4, 5])
    right = dealer.share([-5, 6, -2])
    assert values(reveal(multiply(left, right, dealer))) == [-15, -24, -10]


def test_hand_example():
    network = example()
    plaintext = forward(
        network["x"], network["w1"], network["b1"], network["w2"], network["b2"]
    )
    assert values(plaintext) == [6]

    dealer = Dealer(1)
    opened = reveal(
        secure_forward(
            dealer.share(network["x"]),
            dealer.share(network["w1"]),
            dealer.share(network["b1"]),
            dealer.share(network["w2"]),
            dealer.share(network["b2"]),
            dealer,
        )
    )
    assert values(opened) == [6]
    assert dealer.multiplications == 2 * 2 + 2 + 1 * 2


def test_random_networks_match_plaintext():
    rng = np.random.default_rng(7)
    for seed in range(20):
        features = rng.integers(-4, 5, size=3).tolist()
        weight1 = rng.integers(-4, 5, size=(4, 3)).tolist()
        bias1 = rng.integers(-4, 5, size=4).tolist()
        weight2 = rng.integers(-4, 5, size=(2, 4)).tolist()
        bias2 = rng.integers(-4, 5, size=2).tolist()
        dealer = Dealer(seed)
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
        assert values(opened) == values(plaintext)


if __name__ == "__main__":
    test_share_roundtrip_and_hiding()
    test_beaver_matches_product()
    test_hand_example()
    test_random_networks_match_plaintext()
    print("4 tests passed")
