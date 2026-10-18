# 秘密分享上的两层网络

一个最小 demo：两个互不串通的参与方，在加法秘密分享上完成一次两层网络前向。任意一方单独持有的份额还原不出输入、权重或输出。

```text
h = (W1 @ x + b1) ** 2
y = W2 @ h + b2
```

平方是这个实现里的非线性。它只需要乘法。ReLU 需要比较协议，没有放进这个最小版本。算术在素数域 `2^61-1` 上进行，demo 里的整数很小，结果与普通整数前向一致。

## 运行

```bash
pip install -r requirements.txt
python demo.py
python test_ssnn.py
```

`demo.py` 先跑一个可以手算的例子，明文结果是 `[6]`，再用一个随机小网络对照。

## 协议

- 客户端在本地把输入、两层权重和偏置拆成两份，每个参与方拿一份。
- 加法在本地完成：双方把自己的份额相加。
- 乘法使用 Beaver 三元组。双方只交换掩码后的 `e`、`f`，不交换秘密本身。
- 矩阵乘向量由逐元素乘法和本地求和组成。
- 输出份额交回客户端，只有客户端把两份相加。

初始化方负责产生均匀份额和三元组，之后不参与推理。这里用程序里的 `Dealer` 模拟它。

这是半诚实模型：双方按协议计算，并且不会把份额凑到一起。形状是公开的。打开的 Beaver 掩码也是公开的，但在掩码均匀时不泄漏秘密。某一方如果篡改消息，这个 demo 检测不出来。

## 代码

- `ssnn.py`：域运算、拆分、Beaver 乘法、两层前向
- `demo.py`：手算例子和随机对照
- `test_ssnn.py`：份额隐私、乘法、手算例子、20 组随机网络

## English

Two non-colluding parties evaluate a 2-layer integer network on additive secret shares. Products use Beaver triples. Square is the nonlinearity, because a comparison protocol for ReLU is out of scope. Run `python demo.py` and `python test_ssnn.py`.
