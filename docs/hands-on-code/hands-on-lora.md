---
title: LoRA 原理和 PyTorch 代码实现
date: 2024-11-09T21:56:00
star: true
tag:
  - transformer
  - LLM
category:
  - hands-on-code
description: 用 PyTorch 实现从零实现 LoRA, 理解 LoRA 的原理，主要是为了展示一个 LoRA 实现的细节
publish: true
permalink: /hands-on-code/hands-on-lora.html
---



## 背景

无论是火热的大模型（LLM）还是文生图模型（Stable  Diffusion）微调的时候，都需要大量的GPU显存，个人的显卡上很难实现，因此各种参数高效（Parameter-Efficient）的方法层出不穷，最受大家欢迎的就是 **LoRA**[《LoRA: Low-Rank Adaptation of Large Language Models》](https://papers.cool/arxiv/2106.09685)。

LoRA 有很多的优点，节约显存，训练快，效果损失较小（相对于全参数微调），推理的时候不增加耗时，可以做一个插入式组件使用。缺点当然也有，那就是还是会有一些效果的损失（笑）。

> 减少显存占用的主要原因是训练参数变小了（比如只对 qkv 层做 LoRA）


>
> 不喜欢看文字的同学可以看 [B站视频-chaofa用代码打点酱油](https://www.bilibili.com/video/BV1fHmkYyE2w/),
> 
> 或者视频号：chaofa用代码打点酱油


## 核心原理

核心原理非常的简单，任意一个矩阵 $W_0$，都可以对它进行低秩分解，把一个很大的矩阵拆分成两个小矩矩阵[^1]（$A,B$），在训练的过程中不去改变 $W_0$ 参数，而是去改变 $A B$。具体可以表示为

$$W_{new} = W_0 + AB \tag{1}$$

最终在训练计算的时候是

$$h = W_0x + ABx = (W_0 + AB)x\tag{2}$$

但是一般来说，AB 会进行一定的缩放，使用 $\frac{\alpha}{r}$ 作为缩放因子，所以最终会写成

$$h = (W_0 + \frac{\alpha}{r}AB)x\tag{3}$$

$$\text{s.t.} \quad W_0 \in \mathbb{R}^{n \times m}, \; A \in \mathbb{R}^{n \times r}, \; B \in \mathbb{R}^{r \times m}$$

其中 $r << n \text{ and } r << m$，$r$ 甚至可以设置成 1。


- 为什么说只优化 AB 两个矩阵就可以了呢？这里面的假设是什么？
- $W$ 不是满秩的，里面有大量参数是冗余的，那么其实可以用更接近满秩的矩阵 AB 代替。
> 矩阵都可以表示为若干个线性无关向量，最大的线性无关向量个数就是秩


## PyTorch 代码实现
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LinearLoRALayer(nn.Module):
    def __init__(self, 
        in_features, 
        out_features,
        merge=False,
        rank=8,
        lora_alpha=16,
        dropout=0.1,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.merge = merge
        self.rank = rank

        # linear weight 的 Shape 是 (out_features, in_features), 正确的做法是 xW^T
        self.linear = nn.Linear(in_features, out_features)
        # 这里非常的重要，这里是实现的小细节
        
        if rank > 0:
            # 这里是为了标记 lora_a 和 lora_b 是可训练的参数
            self.lora_a = nn.Parameter(
                torch.zeros(out_features, rank)
            )
            # lora_a 需要初始化为 高斯分布
            # @春风归无期 提醒我 @用代码打点酱油的chaofa : 在调用凯明初始化的时候注释里写的高斯分布，调用的却是均匀分布，而且参数a的值设置的是根号5，但a表示的是leaky relu的负斜率系数，一般是0.01这样的小值，不可能超过1
            nn.init.kaiming_normal_(self.lora_a, a=0.01)

            self.lora_b = nn.Parameter(
                torch.zeros(rank, in_features)
            )
            self.scale = lora_alpha / rank

            # linear 需要设置为不可以训练
            self.linear.weight.requires_grad = False
            self.linear.bias.requires_grad = False
        
        self.dropout = nn.Dropout(
            dropout
        ) if dropout > 0 else nn.Identity()

        # 如果采用 merge 进行推理，
        # 那么会把 lora_a 和 lora_b 两个小矩阵的参数直接放到 linear.weight 中
        if merge:
            self.merge_weight()

    
    def forward(self, X):
        # X shape is (batch, seq_len, in_feature)
        # lora_a 是 out_features * rank
        if self.rank > 0 and not self.merge:
            output = self.linear(X) + self.scale * ( X @ (self.lora_a @ self.lora_b).T )
        elif self.rank > 0 and self.merge:
            output = self.linear(X)
        else:
            output = self.linear(X)
        
        return self.dropout(output)

    def merge_weight(self, ):
        if self.merge and self.rank > 0:
            self.linear.weight.data += self.scale * (self.lora_a @ self.lora_b)
    
    def unmerge_weight(self, ):
        if self.rank > 0:
            self.linear.weight.data -= self.scale * (self.lora_a @ self.lora_b)


# 写一段测试代码
# Test the LoRALinear layer
batch_size = 32
seq_len = 128
in_features = 768
out_features = 512
rank = 8
lora_alpha = 16
dropout = 0.1

# Create a test input
x = torch.randn(batch_size, seq_len, in_features)

# Test regular mode (no merge)
lora_layer = LinearLoRALayer(
    in_features=in_features,
    out_features=out_features,
    rank=rank,
    lora_alpha=lora_alpha,
    dropout=dropout,
    merge=False
)

# Forward pass
output = lora_layer(x)
print(f"Output shape (no merge): {output.shape}")  # Should be [batch_size, seq_len, out_features]

# Test merged mode
lora_layer_merged = LinearLoRALayer(
    in_features=in_features,
    out_features=out_features,
    rank=rank,
    lora_alpha=lora_alpha,
    dropout=dropout,
    merge=True
)

# Forward pass with merged weights
output_merged = lora_layer_merged(x)
print(f"Output shape (merged): {output_merged.shape}")  # Should be [batch_size, seq_len, out_features]

# Test weight merging/unmerging
lora_layer.merge_weight()
output_after_merge = lora_layer(x)
lora_layer.unmerge_weight()
output_after_unmerge = lora_layer(x)

print("Max difference after merge/unmerge cycle:", 
      torch.max(torch.abs(output - output_after_unmerge)).item())


```

- Q: 大模型的 LoRA 实现真的这么简单吗？
- A: 原理是这么简单，但是实际实现过程中因为层很多，会有一些配置，比如 QKV 层做 LoRA 还是 FFN 层做 LoRA，这些都会增加代码的复杂性，但是核心原理就是上面的代码。


## References

[^1]: 这里和PCA,SVD 有一些差别。前者是为了据降维/压缩，后者仅仅是为了学习低秩的矩阵（参数可以更新改变）

感兴趣可以阅读我的其他文章：
- [从 self-attention 到 multi-head self-attention](/hands-on-code/from-self-attention-to-multi-head-self-attention.html)
- [手写 transformer decoder（CausalLM）](/hands-on-code/hands-on-causallm-decoder.html)
- [LLM 大模型训练-推理显存占用分析](/post/llm-train-infer-memoery-usage-calculation.html)
- [手写大模型组件之Group Query Attention，从 MHA，MQA 到 GQA](https://bruceyuan.com/hands-on-code/hands-on-group-query-attention-and-multi-query-attention.html)


## 交个朋友🤣
最后欢迎关注我，基本全网同名 [chaofa用代码打点酱油](https://bruceyuan.com/)
- 公众号： ![chaofa用代码打点酱油](https://bruceyuan.com/llms-zero-to-hero/chaofa-wechat-official-account.png)
- [B站-chaofa用代码打点酱油](https://space.bilibili.com/12420432)
- [YouTube-chaofa用代码打点酱油](https://www.youtube.com/@bbruceyuan)
- [chaofa 的 notion 简介](https://chaofa.notion.site/11a569b3ecce49b2826d679f5e2fdb54)
