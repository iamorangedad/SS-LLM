"""Checks for secret sharing, Beaver multiplication, and the two-layer forward pass."""

import numpy as np

from comm import PARTY_0, PARTY_1, Channel, deserialize, serialize
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


def test_message_arrives_only_at_destination():
    channel = Channel()
    mask_for_b = np.array([7, 9, 11], dtype=object)
    mask_for_a = np.array([4, 5, 6], dtype=object)
    payload_ab = serialize(mask_for_b)
    payload_ba = serialize(mask_for_a)
    channel.send(PARTY_0, PARTY_1, payload_ab, tag="e")
    assert channel.waiting(PARTY_0) == 0
    assert channel.waiting(PARTY_1) == 1
    channel.send(PARTY_1, PARTY_0, payload_ba, tag="f")
    assert channel.waiting(PARTY_0) == 1
    assert channel.waiting(PARTY_1) == 1
    try:
        channel.receive("10.0.0.9")
        raise AssertionError("an address outside the link accepted a receive")
    except ValueError:
        pass

    delivered_to_a = channel.receive(PARTY_0)
    delivered_to_b = channel.receive(PARTY_1)
    assert delivered_to_a.src == PARTY_1
    assert delivered_to_a.payload == payload_ba
    assert delivered_to_b.src == PARTY_0
    assert delivered_to_b.payload == payload_ab
    assert values(deserialize(delivered_to_b.payload, mask_for_b.shape)) == values(mask_for_b)
    assert channel.waiting(PARTY_0) == 0
    assert channel.waiting(PARTY_1) == 0
    try:
        channel.receive(PARTY_0)
        raise AssertionError("party 0 received a message that was not addressed to it")
    except RuntimeError:
        pass

    assert [(entry.src, entry.dst) for entry in channel.log] == [
        (PARTY_0, PARTY_1),
        (PARTY_1, PARTY_0),
    ]
    assert channel.log[0].nbytes == len(payload_ab) == 4 + 8 * mask_for_b.size
    assert channel.log[1].nbytes == len(payload_ba) == 4 + 8 * mask_for_a.size
    assert [entry.tag for entry in channel.log] == ["e", "f"]


def test_secure_forward_logs_both_directions():
    network = example()
    dealer = Dealer(1)
    channel = Channel()
    opened = reveal(
        secure_forward(
            dealer.share(network["x"]),
            dealer.share(network["w1"]),
            dealer.share(network["b1"]),
            dealer.share(network["w2"]),
            dealer.share(network["b2"]),
            dealer,
            channel,
        )
    )
    assert values(opened) == [6]
    assert channel.log
    assert {(entry.src, entry.dst) for entry in channel.log} == {
        (PARTY_0, PARTY_1),
        (PARTY_1, PARTY_0),
    }
    assert {entry.tag for entry in channel.log} == {"e", "f"}
    # Three tensor multiplications (W1 @ x, the square, W2 @ h), two masks, two directions.
    assert len(channel.log) == 12
    assert sum(entry.dst == PARTY_1 for entry in channel.log) == 6
    assert sum(entry.dst == PARTY_0 for entry in channel.log) == 6
    # Wire size is a 4-byte count plus 8 bytes per field element: (2, 2), (2,), (1, 2).
    assert sum(entry.nbytes for entry in channel.log) == 4 * (36 + 20 + 20)


if __name__ == "__main__":
    test_share_roundtrip_and_hiding()
    test_beaver_matches_product()
    test_hand_example()
    test_random_networks_match_plaintext()
    test_message_arrives_only_at_destination()
    test_secure_forward_logs_both_directions()
    print("6 tests passed")
