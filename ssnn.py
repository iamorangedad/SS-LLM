"""Two-layer neural network on two-party additive secret shares.

This is a minimal, checkable inference demo, not a production MPC library.
User documentation: README.md (English, primary) and README_CH.md.

Threat model
    Semi-honest and non-colluding. Each party holds only its own share and
    sees the Beaver masks that multiplication opens.
    Those mask shares are the only values placed on the simulated link in
    comm.py. The link is in-process; it is not a socket and it is not encrypted.
    A trusted dealer samples those masks beforehand and does not take part
    in inference.
    Shares are uniform in the prime field 2^61-1, so one share is
    information-theoretically independent of the secret.
    Input, weights, biases, and output stay split until the client opens
    the output.

Network (integer arithmetic)
    h = (W1 @ x + b1) ** 2
    y = W2 @ h + b2

Square is a nonlinearity that needs only multiplication. ReLU needs a
comparison protocol, which is outside this minimal implementation.
The field is large enough that the demo's small integers do not wrap, so
the result matches an ordinary integer forward pass.
"""

from dataclasses import dataclass

import numpy as np

from comm import PARTY_0, PARTY_1, Channel, deserialize, serialize

PRIME = 2**61 - 1


@dataclass
class Shared:
    """Two shares of one tensor. Party 0 holds s0 and party 1 holds s1."""

    s0: np.ndarray
    s1: np.ndarray

    @property
    def shape(self):
        if self.s0.shape != self.s1.shape:
            raise ValueError("the two shares have different shapes")
        return self.s0.shape


class Dealer:
    """Trusted setup: split secrets and sample multiplication triples only."""

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
    """Encode an integer tensor as field elements. Negatives wrap via Python modulo."""

    array = np.asarray(values, dtype=object)
    encoded = [int(v) % PRIME for v in array.reshape(-1)]
    return np.array(encoded, dtype=object).reshape(array.shape)


def to_signed(value) -> int:
    reduced = int(value) % PRIME
    if reduced > PRIME // 2:
        return reduced - PRIME
    return reduced


def reveal(shared: Shared) -> np.ndarray:
    """Client reconstruction. Parties do not call this during the protocol."""

    total = (shared.s0 + shared.s1) % PRIME
    signed = [to_signed(v) for v in np.asarray(total, dtype=object).reshape(-1)]
    return np.array(signed, dtype=object).reshape(shared.shape)


def _party_mask(share: np.ndarray, triple_share: np.ndarray) -> np.ndarray:
    """Local mask share that this party will open."""

    return (share - triple_share) % PRIME


def _party_product(opened_e, opened_f, a, b, c, add_cross: bool) -> np.ndarray:
    """Finish a local product from public masks and this party's triple share."""

    result = (c + opened_e * b + opened_f * a) % PRIME
    if add_cross:
        result = (result + opened_e * opened_f) % PRIME
    return result


def _exchange_mask(
    channel: Channel, mask0: np.ndarray, mask1: np.ndarray, tag: str
) -> np.ndarray:
    """Each party sends its mask share and opens the mask from what it receives.

    Party 0 adds its local share to the payload addressed to PARTY_0.
    Party 1 does the same with the payload addressed to PARTY_1.
    The secret, the weights, and the other party's raw share are not sent.
    """

    channel.send(PARTY_0, PARTY_1, serialize(mask0), tag)
    channel.send(PARTY_1, PARTY_0, serialize(mask1), tag)
    to_party0 = channel.receive(PARTY_0)
    to_party1 = channel.receive(PARTY_1)
    if to_party0.src != PARTY_1 or to_party0.tag != tag:
        raise RuntimeError("party 0 received an unexpected mask share")
    if to_party1.src != PARTY_0 or to_party1.tag != tag:
        raise RuntimeError("party 1 received an unexpected mask share")
    peer0 = deserialize(to_party0.payload, np.shape(mask0))
    peer1 = deserialize(to_party1.payload, np.shape(mask1))
    opened0 = (np.asarray(mask0, dtype=object) + peer0) % PRIME
    opened1 = (np.asarray(mask1, dtype=object) + peer1) % PRIME
    if [int(v) for v in np.asarray(opened0, dtype=object).flat] != [
        int(v) for v in np.asarray(opened1, dtype=object).flat
    ]:
        raise RuntimeError("the two parties opened different masks")
    return opened0


def multiply(
    left: Shared, right: Shared, dealer: Dealer, channel: Channel | None = None
) -> Shared:
    """Beaver multiplication.

    The only cross-party messages are the mask shares of e and f. Each party
    sends its share to the other address and reconstructs the opened mask
    from its local share plus the share it receives. A fresh in-process link
    is used when `channel` is omitted.
    """

    if left.shape != right.shape:
        raise ValueError("multiplied shares must have the same shape")
    if channel is None:
        channel = Channel()

    shape = left.shape
    raw_a = dealer._uniform(shape)
    raw_b = dealer._uniform(shape)
    raw_c = (raw_a * raw_b) % PRIME
    triple_a = dealer.share(raw_a)
    triple_b = dealer.share(raw_b)
    triple_c = dealer.share(raw_c)
    dealer.multiplications += int(np.prod(shape, dtype=int))

    opened_e = _exchange_mask(
        channel,
        _party_mask(left.s0, triple_a.s0),
        _party_mask(left.s1, triple_a.s1),
        "e",
    )
    opened_f = _exchange_mask(
        channel,
        _party_mask(right.s0, triple_b.s0),
        _party_mask(right.s1, triple_b.s1),
        "f",
    )
    product0 = _party_product(
        opened_e, opened_f, triple_a.s0, triple_b.s0, triple_c.s0, add_cross=True
    )
    product1 = _party_product(
        opened_e, opened_f, triple_a.s1, triple_b.s1, triple_c.s1, add_cross=False
    )
    return Shared(product0, product1)


def add(left: Shared, right: Shared) -> Shared:
    if left.shape != right.shape:
        raise ValueError("added shares must have the same shape")
    return Shared((left.s0 + right.s0) % PRIME, (left.s1 + right.s1) % PRIME)


def square(value: Shared, dealer: Dealer, channel: Channel | None = None) -> Shared:
    return multiply(value, value, dealer, channel)


def matvec(
    weight: Shared, vector: Shared, dealer: Dealer, channel: Channel | None = None
) -> Shared:
    """Matrix-vector product on shares. Both the matrix and the vector are shared."""

    if weight.s0.ndim != 2 or vector.s0.ndim != 1:
        raise ValueError("matvec expects an (out, in) matrix and a 1-D vector")
    rows, cols = weight.shape
    if vector.shape != (cols,):
        raise ValueError("matrix width must equal the vector length")

    expanded = Shared(
        np.broadcast_to(vector.s0, (rows, cols)),
        np.broadcast_to(vector.s1, (rows, cols)),
    )
    products = multiply(weight, expanded, dealer, channel)
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
    channel: Channel | None = None,
) -> Shared:
    """Evaluate the two-layer network on shares without opening intermediates.

    Beaver openings travel on `channel`. When it is omitted, one fresh
    in-process link carries every mask share in this forward pass.
    """

    if channel is None:
        channel = Channel()

    hidden_input = add(matvec(weight1, features, dealer, channel), bias1)
    hidden = square(hidden_input, dealer, channel)
    return add(matvec(weight2, hidden, dealer, channel), bias2)


def forward(features, weight1, bias1, weight2, bias2) -> np.ndarray:
    """Plaintext integer forward pass of the same network, for comparison."""

    features = [int(v) for v in np.asarray(features, dtype=object).reshape(-1)]
    weight1 = np.asarray(weight1, dtype=object)
    bias1 = [int(v) for v in np.asarray(bias1, dtype=object).reshape(-1)]
    weight2 = np.asarray(weight2, dtype=object)
    bias2 = [int(v) for v in np.asarray(bias2, dtype=object).reshape(-1)]

    if weight1.ndim != 2 or weight1.shape[1] != len(features):
        raise ValueError("first-layer weight shape does not match the input")
    if len(bias1) != weight1.shape[0]:
        raise ValueError("first-layer bias length does not match the hidden width")

    hidden = []
    for row, bias in zip(weight1, bias1):
        activation = bias
        for weight, feature in zip(row, features):
            activation += int(weight) * feature
        hidden.append(activation * activation)

    if weight2.ndim != 2 or weight2.shape[1] != len(hidden):
        raise ValueError("second-layer weight shape does not match the hidden width")
    if len(bias2) != weight2.shape[0]:
        raise ValueError("second-layer bias length does not match the output width")

    output = []
    for row, bias in zip(weight2, bias2):
        value = bias
        for weight, activation in zip(row, hidden):
            value += int(weight) * activation
        output.append(value)
    return np.array(output, dtype=object)


def example():
    """Hand-checked example whose plaintext output is [6]. See README.md."""

    return {
        "x": [1, 2],
        "w1": [[1, 0], [1, -1]],
        "b1": [0, 1],
        "w2": [[2, 3]],
        "b2": [4],
    }
