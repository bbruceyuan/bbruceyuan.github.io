---
title: 手写大模型组件之Group Query Attention，从 MHA，MQA 到 GQA
date: 2024-12-08T22:00:00
star: true
tag:
  - transformer
  - LLM
category:
  - hands-on-code
description: 了解注意力机制变体，包括MHA（Multi-Head Attention）、MQA（Multi-Query Attention）和GQA（Group Query Attention）。通过手写代码实现，探讨三种注意力机制的异同，以及GQA在推理性能优化方面的优势。
publish: true
permalink: /hands-on-code/hands-on-group-query-attention-and-multi-query-attention.html
banner: https://bruceyuan.com/img/huggingface.png
---

- GQA（Group Query Attention）的优点：效果损失小，推理的时候可以加速（来自于kvcache小，内存取数少）。
- 仔细阅读 MHA, MQA 和 GQA的区别，就会发现 MHA 和 MQA 都是 GQA 的特殊表达形式
    - 三者可以用同一套代码，只需要修改【GQA】代码里面的 `nums_key_value_head` 参数就可
    - `nums_key_value_head` 设置等于 1 就是 MQA
    - `nums_key_value_head` 设置等于 `nums_head` 就是 MHA


## multi-head self-attention

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

        attention_weight = self.att_dropout(attention_weight)
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


## Group Query Attention
> 备注：以下代码省略了 attention_dropout attention_mask等情况的处理，真实实现过程中需要考虑。
```python
import torch
import torch.nn as nn
import math

# 忽略了 attention_mask, attention_dropout; 
class GroupQueryAttention(nn.Module):
    def __init__(self, hidden_dim, nums_head, nums_key_value_head):
        super().__init__()
        assert hidden_dim % nums_head == 0 # 可以整除
        assert nums_head % nums_key_value_head == 0  # N 个 query head 为一组

        self.hidden_dim = hidden_dim
        self.nums_head = nums_head
        self.nums_key_value_head = nums_key_value_head
        self.head_dim = hidden_dim // nums_head

        # 初始化 qkv o
        self.q_proj = nn.Linear(hidden_dim, nums_head * self.head_dim)  # out feature_size (nums_head * head_dim)
        # k v out shape (nums_key_value_head * head_dim)
        self.k_proj = nn.Linear(hidden_dim, nums_key_value_head * self.head_dim)
        self.v_proj = nn.Linear(hidden_dim, nums_key_value_head * self.head_dim)

        self.o_proj = nn.Linear(hidden_dim, hidden_dim) # input_size nums_head * head_dim

    def forward(self, X, attention_mask=None):
        # X shape (batch, seq, hidden_dim)
        batch_size, seq, _ = X.size()

        # qkv projection
        q = self.q_proj(X)  # （batch, seq, hidden_dim)
        k = self.k_proj(X)
        v = self.v_proj(X) 

        # attention_weight 目标shape 是 (batch, nums_head, seq, seq)
        q = q.view(batch_size, seq, self.nums_head, self.head_dim)
        k = k.view(batch_size, seq, self.nums_key_value_head, self.head_dim)
        v = v.view(batch_size, seq, self.nums_key_value_head, self.head_dim)

        # 关注: nums_head 和 nums_key_value_head 的关系
        q = q.transpose(1, 2) # (b, nums_head, seq, head_dim)
        k = k.transpose(1, 2) # (b, nums_key_value_head, seq, head_dim)
        v = v.transpose(1, 2)  # (b, nums_key_value_head, seq, head_dim)

        # k v repeat； （广播操作）
        k = k.repeat_interleave(self.nums_head // self.nums_key_value_head, dim=1)
        v = v.repeat_interleave(self.nums_head // self.nums_key_value_head, dim=1)

        attention_score = (q @ k.transpose(2, 3)) / math.sqrt(self.head_dim)

        attention_weight = torch.softmax(attention_score, dim=-1)
        # （attention_mask 忽略） # 可以看前面的视频

        output = attention_weight @ v  # (b, nums_head, seq, head_dim)

        # output projection 变成 (b, seq, hidden_dim)
        output = output.transpose(1, 2).contiguous()
        final_output = self.o_proj(output.view(batch_size, seq, -1))

        return final_output

# 测试
x = torch.rand(3, 2, 128)
net = GroupQueryAttention(128, 8, 4)
net(x).shape

```

## Multi Query Attention
由于 MQA 是 GQA 的一种特殊形式，因此只要在参数设置的时候将 nums_key_value_head = 1 就是 Multi Query Self-Attention。


## REF
视频链接：TODO
