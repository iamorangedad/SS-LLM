"""In-process link between two party addresses.

`Channel` copies bytes from one address to the other and records a log.
It does not open sockets, bind ports, or talk to a real network.
Payloads are not encrypted. The threat model stays semi-honest.
"""

from dataclasses import dataclass

import numpy as np

PARTY_0 = "10.0.0.1"
PARTY_1 = "10.0.0.2"


@dataclass(frozen=True)
class Transfer:
    """One simulated delivery: who sent it, who it was for, and how big it was."""

    src: str
    dst: str
    nbytes: int
    tag: str


@dataclass(frozen=True)
class Message:
    """A payload waiting in the destination address's inbox."""

    src: str
    payload: bytes
    tag: str


class Channel:
    """Mailbox for two addresses. Each party receives only its own IP."""

    def __init__(self, left: str = PARTY_0, right: str = PARTY_1):
        if left == right:
            raise ValueError("the two addresses must differ")
        self.left = left
        self.right = right
        self._inbox = {left: [], right: []}
        self.log: list[Transfer] = []

    def send(self, src: str, dst: str, payload: bytes, tag: str = "") -> None:
        """Deliver `payload` to `dst` and append a log entry."""

        self._check(src)
        self._check(dst)
        if src == dst:
            raise ValueError("refusing to send a message to its own source")
        if not isinstance(payload, (bytes, bytearray)):
            raise TypeError("payload must be bytes")
        if not isinstance(tag, str):
            raise TypeError("tag must be a string")
        body = bytes(payload)
        self._inbox[dst].append(Message(src=src, payload=body, tag=tag))
        self.log.append(Transfer(src=src, dst=dst, nbytes=len(body), tag=tag))

    def receive(self, addr: str) -> Message:
        """Return the next message addressed to `addr`.

        The other address's inbox is left untouched.
        """

        self._check(addr)
        inbox = self._inbox[addr]
        if not inbox:
            raise RuntimeError(f"no message addressed to {addr}")
        return inbox.pop(0)

    def waiting(self, addr: str) -> int:
        self._check(addr)
        return len(self._inbox[addr])

    def _check(self, addr: str) -> None:
        if addr not in self._inbox:
            raise ValueError(f"unknown address {addr}")


def serialize(values) -> bytes:
    """Pack integers as a 4-byte count plus one unsigned 8-byte value each."""

    flat = [int(v) for v in np.asarray(values, dtype=object).flat]
    if any(value < 0 or value >= 2**64 for value in flat):
        raise ValueError("value does not fit in an unsigned 8-byte field element")
    body = bytearray(len(flat).to_bytes(4, "big"))
    for value in flat:
        body += value.to_bytes(8, "big")
    return bytes(body)


def deserialize(payload: bytes, shape) -> np.ndarray:
    """Unpack bytes from `serialize` and reshape them to `shape`."""

    if not isinstance(payload, (bytes, bytearray)):
        raise TypeError("payload must be bytes")
    raw = bytes(payload)
    if len(raw) < 4:
        raise ValueError("payload is shorter than its count header")
    count = int.from_bytes(raw[:4], "big")
    expected = 4 + 8 * count
    if len(raw) != expected:
        raise ValueError("payload length does not match the element count")
    values = [
        int.from_bytes(raw[offset : offset + 8], "big") for offset in range(4, expected, 8)
    ]
    array = np.array(values, dtype=object)
    expected_size = int(np.prod(shape, dtype=int))
    if array.size != expected_size:
        raise ValueError("payload element count does not match the tensor shape")
    return array.reshape(shape)


def format_trace(channel: Channel) -> str:
    """English trace of source, destination, byte length, and tag."""

    lines = [
        f"Simulated link {channel.left} <-> {channel.right} (in-process, not a real network)",
        "Cross-party payloads are Beaver mask shares e and f.",
    ]
    for entry in channel.log:
        lines.append(f"  {entry.src} -> {entry.dst}  {entry.nbytes} bytes  {entry.tag}")
    total = sum(entry.nbytes for entry in channel.log)
    lines.append(f"{len(channel.log)} transfers, {total} bytes")
    return "\n".join(lines)
