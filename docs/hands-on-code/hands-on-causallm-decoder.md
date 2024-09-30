---
title: 手写 transformer decoder（CausalLM）
date: 2024-08-18T13:00:00
star: true
tag:
  - transformer
category:
  - hands-on-code
description: 手写一个 Causal Language Model，或者说简化版的 transformer 中的 decoder。
publish: true
permalink: /hands-on-code/hands-on-causallm-decoder.html
banner: https://bruceyuan.com/img/huggingface.png
---

## 阅读须知

面试过程中让写 transformers Decoder 一定要沟通清楚是写一个 CausalLM decoder 还是原版的，原版的比较复杂，一般也不会让写。这里的 Decoder 一般指的是 **CausalLM**，具体变化是少了 encoder 部分的输入，所以也就没有了 encoder and decoder cross attention。

- 因为重点希望写 CausalLM，所以没有 Cross attention 和 也省略了 token embedding 这一步。

## 知识点

- transformers decoder 的流程是：input -> self-attention -> cross-attention -> FFN
- causalLM decoder 的流程是 input -> self-attention -> FFN
  - 其他 `[self-attention, FFN]` 是一个 block，一般会有很多的 block
- FFN 矩阵有两次变化，一次升维度，一次降维度。其中 LLaMA 对于 GPT 的改进还有把 GeLU 变成了 SwishGLU，多了一个矩阵。所以一般升维会从 `4h -> 4h * 2 / 3`
- 原版的 transformers 用 post-norm, 后面 gpt2, llama 系列用的是 pre-norm。其中 llama 系列一般用 RMSNorm 代替 GPT and transformers decoder 中的 LayerNorm。

具体实现：

```python
# 导入相关需要的包
import math
import torch
import torch.nn as nn

import warnings
warnings.filterwarnings(action="ignore")

# 写一个 Block
class SimpleDecoder(nn.Module):
    def __init__(self, hidden_dim, nums_head, dropout=0.1):
        super().__init__()

        self.nums_head = nums_head
        self.head_dim = hidden_dim // nums_head
        self.dropout = dropout

        # 这里按照 transformers 中的 decoder 来写，用 post_norm 的方式实现，主意有 残差链接
        # eps 是为了防止溢出；其中 llama 系列的模型一般用的是 RMSnorm 以及 pre-norm（为了稳定性）
        # RMSnorm 没有一个 recenter 的操作，而 layernorm 是让模型重新变成 均值为 0，方差为 1
        # RMS 使用 w平方根均值进行归一化 $\sqrt{\frac{1}{n} \sum_{1}^{n}{a_i^2} }$
        self.layernorm_att = nn.LayerNorm(hidden_dim, eps=0.00001)

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)
        self.drop_att = nn.Dropout(0.1)

        # for ffn 准备
        self.layernorm_ffn = nn.LayerNorm(hidden_dim, eps=0.00001)

        self.up_proj = nn.Linear(hidden_dim, hidden_dim * 4)
        self.down_proj = nn.Linear(hidden_dim * 4, hidden_dim)
        self.layernorm_ffn = nn.LayerNorm(hidden_dim, eps=0.00001)
        self.act_fn = nn.ReLU()
        self.drop_ffn = nn.Dropout(0.1)

    def attention_output(self, query, key, value, attention_mask=None):
        # 计算两者相关性
        key = key.transpose(2, 3)  # (batch, num_head, head_dim, seq)
        att_weight = torch.matmul(query, key) / math.sqrt(self.head_dim)

        # attention mask 进行依次调整；变成 causal_attention
        if attention_mask is not None:
            # 变成下三角矩阵
            attention_mask = attention_mask.tril()
            att_weight = att_weight.masked_fill(attention_mask == 0, float("-1e20"))
        else:
            # 人工构造一个下三角的 attention mask
            attention_mask = torch.ones_like(att_weight).tril()
            att_weight = att_weight.masked_fill(attention_mask == 0, float("-1e20"))

        att_weight = torch.softmax(att_weight, dim=-1)
        print(att_weight)

        att_weight = self.drop_att(att_weight)

        mid_output = torch.matmul(att_weight, value)
        # mid_output shape is: (batch, nums_head, seq, head_dim)

        mid_output = mid_output.transpose(1, 2).contiguous()
        batch, seq, _, _ = mid_output.size()
        mid_output = mid_output.view(batch, seq, -1)
        output = self.o_proj(mid_output)
        return output

    def attention_block(self, X, attention_mask=None):
        batch, seq, _ = X.size()
        query = self.q_proj(X).view(batch, seq, self.nums_head, -1).transpose(1, 2)
        key = self.k_proj(X).view(batch, seq, self.nums_head, -1).transpose(1, 2)
        value = self.v_proj(X).view(batch, seq, self.nums_head, -1).transpose(1, 2)

        output = self.attention_output(
            query,
            key,
            value,
            attention_mask=attention_mask,
        )
        return self.layernorm_att(X + output)

    def ffn_block(self, X):
        up = self.act_fn(
            self.up_proj(X),
        )
        down = self.down_proj(up)
        # 进行 norm 操作
        return self.layernorm_ffn(X + down)

    def forward(self, X, attention_mask=None):
        # X 一般假设是已经经过 embedding 的输入， (batch, seq, hidden_dim)
        # attention_mask 一般指的是 tokenizer 后返回的 mask 结果，表示哪些样本需要忽略
        # shape 一般是： (batch, nums_head, seq)

        att_output = self.attention_block(X, attention_mask=attention_mask)
        ffn_output = self.ffn_block(att_output)
        return ffn_output


# 测试

x = torch.rand(3, 4, 64)
net = SimpleDecoder(64, 8)
mask = (
    torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0], [1, 1, 1, 0]])
    .unsqueeze(1)
    .unsqueeze(2)
    .repeat(1, 8, 4, 1)
)

net(x, mask).shape
```
