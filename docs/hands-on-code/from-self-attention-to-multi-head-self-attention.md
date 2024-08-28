---
title: 手写 Self-Attention 的四重境界，从 self-attention 到 multi-head self-attention
date: 2024-08-18T12:00:00
star: true
tag:
  - transformer
category:
  - hands-on-code
description: 在 AI 相关的面试中，经常会有面试官让写 self-attention，但是 transformer 这篇文章其实包含很多的细节，因此可能面试官对于 self-attention 实现到什么程度是有不同的预期。因此这里想通过写不同版本的 self-attention 实现来达到不同面试官的预期，四个不同的版本，对应不同的细节程度。
publish: true
permalink: /hands-on-code/from-self-attention-to-multi-head-self-attention.html
banner: https://bruceyuan.com/img/huggingface.png
---

## 背景
在 AI 相关的面试中，经常会有面试官让写 self-attention，但是因为 [transformer](https://arxiv.org/pdf/1706.03762) 这篇文章其实包含很多的细节，因此可能面试官对于 self-attention 实现到什么程度是有不同的预期。因此这里想通过写不同版本的 self-attention 实现来达到不同面试官的预期。以此告诉面试官，了解细节，但是处于时间考虑，可能只写了简化版本，如果有时间可以把完整的写出来。

## Self-Attention

MultiHead Attention 的时候下一章介绍；先熟悉当前这个公式。
### Self Attention 的公式
$$SelfAttention(X) = softmax(\frac{Q\cdot K}{\sqrt{d}}) \cdot V$$
$Q = K = V = W * X$，其中Q K V 对应不同的矩阵 W

### 补充知识点
1. matmul 和 @ 符号是一样的作用
2. 为什么要除以 $\sqrt{d}$？ a. 防止梯度消失 b. 为了让 QK 的内积分布保持和输入一样
3. 爱因斯坦方程表达式用法：` torch.einsum("bqd,bkd-> bqk", X, X).shape`
4. X.repeat(1, 1, 3)  表示在不同的维度进行 repeat操作，也可以用 tensor.expand 操作

### 第一重: 简化版本
- 直接对着公式实现， $SelfAttention(X) = softmax(\frac{Q\cdot K}{\sqrt{d}}) \cdot V$
```python
# 导入相关需要的包
import math
import torch
import torch.nn as nn

import warnings
warnings.filterwarnings(action="ignore")


class SelfAttV1(nn.Module):
    def __init__(self, hidden_dim):
        super(SelfAttV1, self).__init__()
        self.hidden_dim = hidden_dim
        # 一般 Linear 都是默认有 bias
        # 一般来说， input dim 的 hidden dim
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, X):
        # X shape is: (batch, seq_len, hidden_dim)， 一般是和 hidden_dim 相同
        # 但是 X 的 final dim 可以和 hidden_dim 不同
        Q = self.query_proj(X)
        K = self.key_proj(X)
        V = self.value_proj(X)

        # shape is: (batch, seq_len, seq_len)
        # torch.matmul 可以改成 Q @ K.T
        # 其中 K 需要改成 shape 为： (batch, hidden_dim, seq_len)
        attention_value = torch.matmul(Q, K.transpose(-1, -2))
        attention_wight = torch.softmax(
            attention_value / math.sqrt(self.hidden_dim), dim=-1
        )
        # print(attention_wight)
        # shape is: (batch, seq_len, hidden_dim)
        output = torch.matmul(attention_wight, V)
        return output


X = torch.rand(3, 2, 4)
net = SelfAttV1(4)
net(X)
```

### 第二重: 效率优化
- 上面那哪些操作可以合并矩阵优化呢？
	- QKV 矩阵计算的时候，可以合并成一个大矩阵计算。
> 但是当前 `transformers` 实现中，其实是三个不同的 Linear 层

```python
class SelfAttV2(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.dim = dim
        # 这样可以进行加速, 那么为什么现在 Llama, qwen, gpt 等
        self.proj = nn.Linear(dim, dim * 3)

        self.output_proj = nn.Linear(dim, dim)

    def forward(self, X):
        # X shape is: (batch, seq, dim)

        QKV = self.proj(X)  # (batch, seq, dim * 3)
        # reshape 从希望的 q, k, 的形式
        Q, K, V = torch.split(QKV, self.dim, dim=-1)

        # print(x)
        att_weight = torch.softmax(
            Q @ K.transpose(-1, -2) / math.sqrt(self.dim), dim=-1
        )
        output = att_weight @ V
        return self.output_proj(output)


X = torch.rand(3, 2, 4)
net = SelfAttV2(4)
net(X).shape
```

### 第三重: 加入细节
- 看上去 self attention 实现很简单，但里面还有一些细节，还有哪些细节呢？
	- attention 计算的时候有 dropout，而且是比较奇怪的位置
	- attention 计算的时候一般会加入 attention_mask，因为样本会进行一些 padding 操作；
	- MultiHeadAttention 过程中，除了 QKV 三个矩阵之外，还有一个 output 对应的投影矩阵，因此虽然面试让你写 SingleHeadAttention，但是依然要问清楚，是否要第四个矩阵？

```python
class SelfAttV3(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.dim = dim
        # 这样可以进行加速
        self.proj = nn.Linear(dim, dim * 3)
        # 一般是 0.1 的 dropout，一般写作 config.attention_probs_dropout_prob
        # hidden_dropout_prob 一般也是 0.1
        self.att_drop = nn.Dropout(0.1)

        # 不写这个应该也没人怪，应该好像是 MultiHeadAttention 中的产物，这个留给 MultiHeadAttention 也没有问题；
        self.output_proj = nn.Linear(dim, dim)

    def forward(self, X, attention_mask=None):
        # attention_mask shape is: (batch, seq)
        # X shape is: (batch, seq, dim)

        QKV = self.proj(X)  # (batch, seq, dim * 3)
        # reshape 从希望的 q, k, 的形式
        Q, K, V = torch.split(QKV, self.dim, dim=-1)

        att_weight = Q @ K.transpose(-1, -2) / math.sqrt(self.dim)
        if attention_mask is not None:
            # 给 weight 填充一个极小的值
            att_weight = att_weight.masked_fill(attention_mask == 0, float("-1e20"))

        att_weight = torch.softmax(att_weight, dim=-1)

        # 这里在 BERT中的官方代码也说很奇怪，但是原文中这么用了，所以继承了下来
        # （用于 output 后面会更符合直觉？）
        att_weight = self.att_drop(att_weight)

        output = att_weight @ V
        ret = self.output_proj(output)
        return ret


X = torch.rand(3, 4, 2)
b = torch.tensor(
    [
        [1, 1, 1, 0],
        [1, 1, 0, 0],
        [1, 0, 0, 0],
    ]
)
print(b.shape)
mask = b.unsqueeze(dim=1).repeat(1, 4, 1)

net = SelfAttV3(2)
net(X, mask).shape
```

### 面试写法 （完整版）--注意注释

```python
# 导入相关需要的包
import math
import torch
import torch.nn as nn

import warnings

warnings.filterwarnings(action="ignore")

class SelfAttV4(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.dim = dim

        # 这样很清晰
        self.query_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)
        # 一般是 0.1 的 dropout，一般写作 config.attention_probs_dropout_prob
        # hidden_dropout_prob 一般也是 0.1
        self.att_drop = nn.Dropout(0.1)

        # 可以不写；具体和面试官沟通。
        # 这是 MultiHeadAttention 中的产物，这个留给 MultiHeadAttention 也没有问题；
        self.output_proj = nn.Linear(dim, dim)

    def forward(self, X, attention_mask=None):
        # attention_mask shape is: (batch, seq)
        # X shape is: (batch, seq, dim)

        Q = self.query_proj(X)
        K = self.key_proj(X)
        V = self.value_proj(X)

        att_weight = Q @ K.transpose(-1, -2) / math.sqrt(self.dim)
        if attention_mask is not None:
            # 给 weight 填充一个极小的值
            att_weight = att_weight.masked_fill(attention_mask == 0, float("-1e20"))

        att_weight = torch.softmax(att_weight, dim=-1)
        print(att_weight)

        # 这里在 BERT中的官方代码也说很奇怪，但是原文中这么用了，所以继承了下来
        # （用于 output 后面会更符合直觉？）
        att_weight = self.att_drop(att_weight)

        output = att_weight @ V
        ret = self.output_proj(output)
        return ret


X = torch.rand(3, 4, 2)
b = torch.tensor(
    [
        [1, 1, 1, 0],
        [1, 1, 0, 0],
        [1, 0, 0, 0],
    ]
)
print(b.shape)
mask = b.unsqueeze(dim=1).repeat(1, 4, 1)

net = SelfAttV4(2)
net(X, mask).shape
```

## MultiHead-Self-Attention
怎么手写一个 Single Head Self-Attention，但是一般在实际上的训练过程中都会使用 Multi Head, 而且其实也仅仅是 每个 Head 做完 Self-Attention 得到结果之后，进行拼接，然后过一个 output 投影矩阵。

### 第四重：multi-head self-attention
```python
import math
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, nums_head) -> None:
        super().__init__()
        self.nums_head = nums_head

        # 一般来说，
        self.head_dim = hidden_dim // nums_head
        self.hidden_dim = hidden_dim

        # 一般默认有 bias，需要时刻主意，hidden_dim = head_dim * nums_head，所以最终是可以算成是 n 个矩阵
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)

        # gpt2 和 bert 类都有，但是 llama 其实没有
        self.att_dropout = nn.Dropout(0.1)
        # 输出时候的 proj
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, X, attention_mask=None):
        # 需要在 mask 之前 masked_fill
        # X shape is (batch, seq, hidden_dim)
        # attention_mask shape is (batch, seq)

        batch_size, seq_len, _ = X.size()

        Q = self.q_proj(X)
        K = self.k_proj(X)
        V = self.v_proj(X)

        # shape 变成 （batch_size, num_head, seq_len, head_dim）
        q_state = Q.view(batch_size, seq_len, self.nums_head, self.head_dim).permute(
            0, 2, 1, 3
        )
        k_state = K.view(batch_size, seq_len, self.nums_head, self.head_dim).transpose(
            1, 2
        )
        v_state = V.view(batch_size, seq_len, self.nums_head, self.head_dim).transpose(
            1, 2
        )
        # 主意这里需要用 head_dim，而不是 hidden_dim
        attention_weight = (
            q_state @ k_state.transpose(-1, -2) / math.sqrt(self.head_dim)
        )
        print(type(attention_mask))
        if attention_mask is not None:
            attention_weight = attention_weight.masked_fill(
                attention_mask == 0, float("-1e20")
            )

        # 第四个维度 softmax
        attention_weight = torch.softmax(attention_weight, dim=3)
        print(attention_weight)

        output_mid = attention_weight @ v_state

        # 重新变成 (batch, seq_len, num_head, head_dim)
        # 这里的 contiguous() 是相当于返回一个连续内存的 tensor，一般用了 permute/tranpose 都要这么操作
        # 如果后面用 Reshape 就可以不用这个 contiguous()，因为 view 只能在连续内存中操作
        output_mid = output_mid.transpose(1, 2).contiguous()

        # 变成 (batch, seq, hidden_dim),
        output = output_mid.view(batch_size, seq_len, -1)
        output = self.o_proj(output)
        return output


attention_mask = (
    torch.tensor(
        [
            [0, 1],
            [0, 0],
            [1, 0],
        ]
    )
    .unsqueeze(1)
    .unsqueeze(2)
    .expand(3, 8, 2, 2)
)

x = torch.rand(3, 2, 128)
net = MultiHeadAttention(128, 8)
net(x, attention_mask).shape
```

> 这里再次解释一下，为什么现在现在的代码实现都是 q k v 的投影矩阵都是分开写的，这是因为现在的模型很大，本身可能会做 张量并行，流水线并行等方式，所以分开写问题也不大（分开写很清晰），可能是加速效果并不明显。