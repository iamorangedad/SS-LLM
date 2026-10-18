# A Two-Layer Network on Secret Shares

[中文文档](README_CH.md)

A minimal demo. Two non-colluding parties evaluate one forward pass of a two-layer network on additive secret shares. A single share does not reveal the input, the weights, or the output.

```text
h = (W1 @ x + b1) ** 2
y = W2 @ h + b2
```

Square is the nonlinearity in this implementation. It needs only multiplication. ReLU needs a comparison protocol, which this minimal version leaves out. Arithmetic is in the prime field `2^61-1`. The integers in the demo are small, so the field result matches an ordinary integer forward pass.

## Run

```bash
pip install -r requirements.txt
python demo.py
python test_ssnn.py
```

`demo.py` runs a hand-checked example whose plaintext result is `[6]`, then checks a random small network.

## Protocol

- The client splits the input, both weight matrices, and both biases locally. Each party receives one share.
- Addition is local. Each party adds its own shares.
- Multiplication uses Beaver triples. The parties exchange only the masked values `e` and `f`, not the secrets.
- A matrix-vector product is element-wise multiplications followed by a local sum.
- Output shares go back to the client. Only the client adds them.

A dealer samples the uniform shares and the triples, then leaves. `Dealer` in `ssnn.py` stands in for that setup.

The threat model is semi-honest. Both parties follow the protocol and do not pool their shares. Tensor shapes are public. Opened Beaver masks are public too, and they do not reveal the secret while the masks are uniform. This demo does not detect a party that sends malformed messages.

## Code

- `ssnn.py`: field arithmetic, sharing, Beaver multiplication, and the two-layer forward pass
- `demo.py`: the hand-checked example and a random comparison
- `test_ssnn.py`: share privacy, multiplication, the hand-checked example, and 20 random networks

## Hand-checked example

```text
x  = [1, 2]
W1 = [[1, 0], [1, -1]]
b1 = [0, 1]
W2 = [[2, 3]]
b2 = [4]

z = [1*1 + 0*2, 1*1 + (-1)*2 + 1] = [1, 0]
h = [1, 0]
y = [2*1 + 3*0 + 4] = [6]
```

The secure forward pass opens to the same `[6]`.
