---
title: LLM activate function激活函数的进化之路，从 ReLU，GELU 到 SwiGLU(swishGLU)
date: 2025-01-27T18:58:00
star: true
tag:
  - transformer
  - LLM
category:
  - hands-on-code
  - llms-zero-to-hero
description: "主要介绍了从基础的 ReLU 到 GELU，再到现代大语言模型中广泛使用的 SwishGLU 的发展过程, 介绍了深度学习中激活函数演进历程。文章详细讲解了各个激活函数的数学原理和实现方式，并重点分析了 SwishGLU 如何结合 Swish 激活函数和 GLU 门控单元的优点。同时，文章还提供了完整的 PyTorch 代码实现，展示了如何在神经网络中使用这些激活函数，特别是在大语言模型的 FFN（前馈神经网络）层中的应用。对于想要深入理解现代深度学习模型架构的开发者和研究者来说，这是一份很有价值的参考资料。"
publish: true
permalink: /llms-zero-to-hero/activate-function-from-relu-gelu-to-swishglu.html
banner: https://yuanchaofa.com/img/huggingface.png
---

## 1. 背景

自 chatGPT 22年底问世以来，大模型（Large Language Model, LLM）一般使用 [Causal Language Model](https://yuanchaofa.com/hands-on-code/hands-on-causallm-decoder.html) 的形式，属于 Transformers 中的 Decoder 部分，其中在 Decoder 的 Block 中有一个 FFN(FeadForward) 层，一般认为这部分参数用于存储知识。而标准的 FFN 一般有一个升维度和降维度的过程，一共有两个权重矩阵，用公式表示为

$$FFN(x) = ReLU(xW_1 + b1)W2 + b2  \tag{1}$$

其中 x shape 是 $(b, s, h)$，w1 shape 是 $(h, 4h)$，w2 shape 是 $(4h, h)$, w1 是升维（up），w2 是降维(down)

激活函数主要是为了实现神经网络学习输入和输出之间的复杂非线性关系而使用的一个函数。在公式 (1) 中，`ReLU` 是一个激活函数（Transfromers原版），可以替换成其他的激活函数，比如 BERT 开始用 `Gaussian Error Linear Unit，GELU` 比较多，随后就成了激活函数的主流选择，但是随着大模型的爆火以及 PaLM 模型的发布，大家开始慢慢使用 swishGLU 作为激活函数，并且作为一个主要的优化点。

具体可以看下面一段代码即可清楚的理解 FFN 模型是什么实现的。

```python
class FeedForward(nn.Module):
    # 实际上就是 MLP
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
             # 激活函数
             nn.ReLU(),  
             #  可以替换成 nn.GELU(),  
             #  但是 如果是 SwishGLU 则实现方式有所不同，接下来就会介绍 swishGLU 是怎么实现的
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout)
        )
    
    def forward(self, x):
        return self.net(x)
```

## 2. 升级之路

### 1. ReLU

ReLU 深度学习以来最常用的激活函数，其公式非常的简单。
$$ReLU(x) = max(0, x) \tag{2}$$

### 2. GELU

从 GPT、BERT 以来，GELU 似乎成了新时代取代 ReLU 的激活函数，具体形式如下：
$$
GELU(x) = x  P(X \le x) = x  \Phi(x)  \tag{3}
$$

其中 $\Phi(x)$ 是标准正态分布的累计分布函数，定义为
$$
\Phi(x) = \frac{1}{2}(1 + erf(\frac{x}{\sqrt{2}}))  \tag{4}
$$
这里的 `erf` 是误差函数
$$
erf(x) = \frac{2}{\sqrt{\pi}} \int_0^x e^{-t^2} dt  \tag{5}
$$

但是这个函数由于计算成本较高，因此有两个初等函数作为近似计算（但目前【2025年1月27日】其实很多框架已经可以精确计算 erf 函数）。

近似计算分析详细可以参见苏神的文章，[GELU的两个初等函数近似是怎么来的](https://spaces.ac.cn/archives/7309)

### 3. SwiGLU（SwishGLU）

SwiGLU（或者swishGLU，以下可能混用） 是 swish 激活函数和 GLU 门控单元的结合体，因此需要分别介绍两者的不同。

其中需要注意的是：在 T5 开始，很多模型（比如 PaLM ）在FFN层都不用 bias 了，也就是说 FFN的公式变成了
$$
FFN(x) = \text{ActiveFunction}(xW_1)W2 \tag{6}
$$

注意公式 6 和公式 1 的区别，一共没有 bias 一个有 bias，但具体得看不同模型的实现，并不能一概而论。

#### 3.1 swish 激活函数

swish 是一个非线性函数（激活函数都是如此，笑🤣），具体公式为：

$$
\text{Swish} = x \sigma(\beta x)
$$
其中 $\beta$ 是一个超参数，当 $\beta = 1$ 时，Swish 就变成了 SiLU (Sigmoid Linear Unit)，大多数框架的默认实现（如 PyTorch、TensorFlow 的 `nn.SiLU()`）使用的是 $\beta = 1$ 的固定版本。

因此如果采用 swish 激活函数，FFN 的公式变成了

$$
FFN(W_1, W_2, x) = \text{Swish}(xW_1)W2
$$

共有两个可学习的矩阵，其中 $w_1,(h, 4h)$ 是升维矩阵，$w_2,(4h, h)$ 是降低维度的矩阵。

#### 3.2 GLU 门控单元

GLU，Gated Linear Units，是一种门控结构（有参数，因此相对于普通的激活函数多了一个 `gate` 矩阵），通过 sigmoid 控制不同维度的激活。公式如下[^1]：

$$
GLU(W, x, V, b, c) = (Wx + b) \otimes \text{sigmoid}(Vx + c)  \tag{7}
$$

这里是不是熟悉 LSTM, GRU 的同学一下就理解，其中需要注意的是，`b, c` 对应的 bias 不是必须的。

对比公式 7 和公式 9，公式 9 中的 $w_{up}$ 对应 公式 7 中的 $W$，而 $w_{gate}$ 对应公式 7 中的 $V$ 矩阵。

#### 3.3 SwiGLU 的表达形式

而 `SwiGLU` 就是把门控函数替换成了 `swish`，并且去除掉了 `bias` 部分，以及把 FFN 层的一个 Linear 层替换成了 GLU 层，因此一共有三个可训练的参数矩阵, w1, w2, w3。

因此最终的公式表达为，

$$
FFN(W_1, W_2, W_3, x) = W_2 \cdot (W_1x \otimes \text{Swish}(W_3x))  \tag{8}
$$

而我们都知道 FFN 是一个升高维度，然后降低维度的过程，因此可以写成，W2 是一个降低维度的参数，W1 是升高维度的过程，而 W3 是一个 Gate 需要用到的参数矩阵。
$$
FFN(w_{up}, w_{down}, w_{gate}) = w_{down} \cdot (w_{up}x \otimes \text{Swish}(w_{gate}x))  \tag{9}
$$

通过这个公式整体就非常的清晰理解使用 swiGLU 的 FFN。

而我们都知道在 basic 版本的 FFN，见公式（1）， 只有 $w_{up}$ 和 $w_{down}$ 分别是 (h, 4h) 和（4h, h），因此整体参数是 $8h^2$。

而公式9 中，一共有三个矩阵，如果想要实现总参数 $8h^2$，那么每一个参数矩阵的大小应该是 $\frac{8h^2}{3}$，因此 $w_{up}, w_{gate}$ 的shape应该是 $(h, \frac{8h}{3})$，$w_{down}$ 的 shape 是 $(\frac{8h}{3}, h)$。

假设输入的 `hidden_dim` 大小是 `hidden_dim`，那么中间层（up 后的维度）大小是 `mid_dim`， 具体计算逻辑如下：

```python
mid_dim = int(8 * hidden_dim / 3)
# multiple_of：make SwiGLU hidden layer size multiple of large power of 2
mid_dim = multiple_of * ((mid_dim + multiple_of - 1) // multiple_of)

# multiple_of 一般设置为 256， LLaMA 和 GPT等模型
```

注意，在 LLM (大语言模型) 架构中，multiple_of 是一个用于优化计算效率的参数，通常设置为 256 或其他 2 的幂次方数（如 128、512 等），最终让 `mid_dim` 调整为 `multiple_of` 的整数倍。这样做有几个原因：

1. 硬件优化：现代 GPU/TPU 在处理 2 的幂次方大小的张量时效率最高
2. 内存对齐：确保内存对齐可以提高计算速度
3. 并行计算效率：某些并行计算操作在处理规整的数字时效率更高

## 3. 带有 swishGLU 的 FFN 代码实现

```python
class FFNExpert(nn.Module):
    def __init__(self, hidden_dim, dropout):   # LLM 进化之路， FFN 激活函数从 GELU -> SwiGLU
        super().__init__()  

        # 有一个 magic number 叫做 8/3
        hidden_dim = hidden_dim
        # 这里可以自己去优化成 multiple_of 的倍数
        mid_dim = hidden_dim * 8 // 3

        self.up = nn.Linear(hidden_dim, mid_dim, bias=False)
        self.down = nn.Linear(mid_dim, hidden_dim, bias=False)
        self.gate = nn.Linear(hidden_dim, mid_dim, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.dropout(
            self.down(
                # up 之后的 Shape 是(b, s, mid_dim)
                # gate 和 up 之后的Shape都是 (b, s, mid_dim)
                # 两者是 element-wise 相乘
                F.silu(
                    self.gate(x)
                ) * self.up(x)
            )
        )
        return out
```

## 参考

- [GELU的两个初等函数近似是怎么来的](https://kexue.fm/archives/7309)
- 非常参考阅读文章：[GLU 和 SwiGLU](https://mingchao.wang/1fb1JNJ6/) 可以写的时候没发现

[^1]: <https://zhuanlan.zhihu.com/p/693332639>

最后欢迎关注我，基本全网同名 [chaofa用代码打点酱油](https://yuanchaofa.com/)

- 公众号： ![chaofa用代码打点酱油](https://yuanchaofa.com/llms-zero-to-hero/chaofa-wechat-official-account.png)
- [B站-chaofa用代码打点酱油](https://space.bilibili.com/12420432)
- [YouTube-chaofa用代码打点酱油](https://www.youtube.com/@bbruceyuan)
- [chaofa 的 notion 简介](https://chaofa.notion.site/11a569b3ecce49b2826d679f5e2fdb54)
