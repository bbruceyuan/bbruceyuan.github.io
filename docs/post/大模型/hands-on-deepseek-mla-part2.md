---
title: MLA(2)：从代码和公式角度理解 DeepSeek MLA 的矩阵吸收 (Projection Absorption)
date: 2025-03-16 21:04:20
tag:
  - transformer
  - LLM
category:
  - hands-on-code
description: 从代码角度深入理解 DeepSeek MLA 算法。从代码角度详细解析 MLA（Multi-head Latent Attention）算法的核心思想，如何通过矩阵吸收来优化 KV Cache。
publish: true
permalink: /post/hands-on-deepseek-mla-projection-absorption.html
---

## 基础原理

这里假设读者对于 MLA有一定的了解，只是不清楚 MLA 算法的实现，关于原版的 MLA 具体实现可以见 [从代码角度学习和彻底理解 DeepSeek MLA 算法](https://bruceyuan.com/post/hands-on-deepseek-mla.html)，视频解读见：[ 完全从零实现DeepSeek MLA算法(MultiHead Latent Attention)-（无矩阵吸收版）](https://www.bilibili.com/video/BV19aP1epEUc)


![deepseek-mla-矩阵吸收之迷-20250316140034131](https://cfcdn.bruceyuan.com/blog/2025/deepseek-mla-矩阵吸收之迷-20250316140034131.webp)

上面的公式详细的解释了MLA 的计算过程，但这是为了后续代码讲解矩阵吸收回顾使用。

> 欢迎关注我的 github repo: [LLMs-zero-to-hero](https://github.com/bbruceyuan/LLMs-Zero-to-Hero)


## CacheDecompressed (CD)

在原始的官方 huggingface 的[实现](https://huggingface.co/deepseek-ai/DeepSeek-V2-Chat/tree/main)中（852行开始），kv cache 缓存的是完整的 kv cache，也就是**升维之后**且应用了 RoPE 位置编码的 kv，而不是压缩后的 $C_t^{KV}$。具体实现见：

```python
def forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
):
    ... 
    # 注意这里的 compressed_kv 是计算出来的
    # 实际只要缓存这个就行，不行看是 kv states
    compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
    # 此处compressed_kv 对应公式中的 c_t^{KV}
    compressed_kv, k_pe = torch.split(
        compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
    )
    ...
        
    # key shape is: (batch, seq_len, num_head, nope_dim + rope_dim)
    key_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
    key_states[:, :, :, : self.qk_nope_head_dim] = k_nope
    key_states[:, :, :, self.qk_nope_head_dim :] = k_pe
    # value shape is (batch, seq_len, num_head, value_dim)
    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )
    ...
```

> 注意代码中 shape 的注释，通过 shape 可以了解缓存的完整的 kv cache

## Cache Compressed_kv (CC)


```python
# CacheCompressed
def forward(self, hidden_states_q: torch.Tensor, q_position_ids: torch.LongTensor, compressed_kv: torch.Tensor):
    ...
    kv_seq_len = compressed_kv.size(1)
    compressed_kv, k_pe = torch.split(
        compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
    )
    k_pe = k_pe.view(bsz, kv_seq_len, 1, self.qk_rope_head_dim).transpose(1, 2)
    kv = self.kv_b_proj(compressed_kv) \
        .view(bsz, kv_seq_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim) \
        .transpose(1, 2)
    
    k_nope, value_states = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
    ... 
```

注意看两者入参的区别，和明显 compressed_kv 明显小于完整的 kv，因为 compressed_kv 是所有 **head 共享**的，缓存的内容从 `(batch, seq_len, num_head, nope_dim + rope_dim)`变成了 `(batch, seq_len, compressed_dim)`。明显减少了缓存内容大小。

但是很明显，这种方式会极大的增加计算量，因为你要从 compressed_dim 升维到正常计算 attention 的 dim，以及需要扩展到 num_head，因为所有 head 是共用 compressed_dim。

## 矩阵吸收 Absorbed_CacheCompressed (A_CC)

图一 37 - 47 公式中， $W^{UK}$ 可以被 $W^{UQ}$ 吸收，而 $W^{UV}$ 可以被 $W^{O}$ 吸收。具体的做法其实也比较简单，是矩阵乘法的变化。
### key 的矩阵吸收

$$
\begin{aligned}
{attention weight} = q_t^\top k_t  \\
= (W^{UQ}c_t^Q)^\top W^{UK}c_t^{KV} \\
= c_t^{Q\top}W^{UQ\top}W^{UK}c_t^{KV} \\
= (c_t^{Q\top}W^{UQ\top}W^{UK})c_t^{KV}
\end{aligned}
$$
解释一个公式，我们计算 attention weight 是通过 $q_t$ 和 $k_t$ 的相似度计算出来的，$t$ 表示某一个 token。其中 $q_t$ 和 $k_t$ 分别是通过 $W^{UQ}$ 和 $W^{UK}$ 升维得到的，但是将转置 $\top$ 拆开后，得到第三个等号的公式，其中 $W^{UQ\top}$ 和 $W^{UK}$ 可以结合，因此在 inference 的时候，我们只需要计算 $(c_t^{Q\top}W^{UQ\top}W^{UK})$ 中的结果就行，而因为 kv cache 的存在，q 在 decoder 的时候，长度就是 1，因此极大的减少了浮点数的运算。

### value 的矩阵吸收

公式和 key 的矩阵吸收比较类似。如果用公式表示为：

$$
\begin{aligned}
{final output} = A_{weight}(W^{UV}C_t^{KV})W^O  \\
= A_{weight}C_t^{KV}W^{UV}W^O \\
= A_{weight}C_t^{KV}(W^{UV}W^O)
\end{aligned}
$$
其中 $W^{UV}W^O$ 可以像 key 中的 $W^{UQ\top}W^{UK}$ 一样被吸收。

但一般来说，其实就是通过调整运算顺序来减少中间大矩阵的生成，用 einsum 表示如下：

```python
v_t = einsum('hdc,blc->blhd', W_UV, c_t_KV) # (1)
o   = einsum('bqhl,blhd->bqhd', a, v_t)     # (2)
u   = einsum('hdD,bhqd->bhD', W_o, o)       # (3)

# 将上述三式合并，得到总的计算过程
u   = einsum('hdc,blc,bqhl,hdD->bhD', W_UV, c_t_KV, a, W_o)

# 利用结合律改变计算顺序
o_  = einsum('bhql,blc->bhqc', a, c_t_KV) # (4)  
# # 相对于 1 来说，中间变量更小，从 (b, l, h, d) 变成了(b, h, q, c)
o   = einsum('bhqc,hdc->bhqd', o_, W_UV)  # (5)
u   = einsum('hdD,bhqd->bqD', W_o, o)     # (6)
```

解释一下上面的变量：
```
h: head_number
d: value dim
c: compressed_dim
l: seq_len
q: seq_len
D: output_dim/hidden_dim
```
### Move Elision (A_CC_ME)
> Absorbed_CacheCompressed_MoveElision (A_CC_ME)


上面的策略会产生大量无用的数据拷贝和广播，同时也会占用大量显存空间导致OOM。可以采用MoveElision优化策略， 即省略此处的拼接RoPE部分和非RoPE部分的过程，而是直接分别计算量部分的额Attention Score并相加（考虑 $q_t^\top k_t = {q_t^C}^\top k_t^C + {q_t^R}^\top k_t^R$）。此段内容来自于：[optimizing-mla](https://github.com/madsys-dev/deepseekv2-profile/blob/main/workspace/blog/optimizing-mla.md)。

```python
# Absorbed_CacheCompressed_MoveElision
def forward(...):
    ...
    # qk_head_dim = self.kv_lora_rank + self.qk_rope_head_dim
    # query_states = k_pe.new_empty(bsz, self.num_heads, q_len, qk_head_dim)
    # query_states[:, :, :, : self.kv_lora_rank] = torch.einsum('hdc,bhid->bhic', q_absorb, q_nope)
    # query_states[:, :, :, self.kv_lora_rank :] = q_pe

    # key_states = k_pe.new_empty(bsz, self.num_heads, kv_seq_len, qk_head_dim)
    # key_states[:, :, :, : self.kv_lora_rank] = compressed_kv.unsqueeze(1)
    # key_states[:, :, :, self.kv_lora_rank :] = k_pe

    # attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.softmax_scale

    attn_weights = torch.matmul(q_pe, k_pe.transpose(2, 3)) + torch.einsum('bhqc,blc->bhql', q_nope, compressed_kv)
    attn_weights *= self.softmax_scale
    ...
```

> 备注：这里的主要区别就是 rope 部分和 nope 部分分开计算 Attention，算完之后两者加起来。

### 最终实现
```python
"""
这是带有矩阵吸收的版本
"""


class MLAV2(nn.Module):
    def __init__(
        self,
        config,
    ):
        super().__init__()

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads

        self.max_postion_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta

        # 对应 query 压缩的向量， 在 deepseek v3 中， hidden_size 7168
        # 但是压缩后的 kv d_c= 512，压缩比例 1/14
        # q 的压缩为 1536 压缩比例 1/4.7
        # rope 部分是 64

        self.q_lora_rank = config.q_lora_rank
        # 对应 query 和 key 进行 rope 的维度
        self.qk_rope_head_dim = config.qk_rope_head_dim
        # 对应 value 压缩的向量
        self.kv_lora_rank = config.kv_lora_rank

        # 对应 每一个 Head 的维度大小
        self.v_head_dim = config.v_head_dim

        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim

        self.q_down_proj = nn.Linear(
            self.hidden_size,
            self.q_lora_rank,
            bias=config.attention_bias,
        )
        self.q_down_layernorm = DeepseekV2RMSNorm(self.q_lora_rank)

        self.q_up_proj = nn.Linear(
            self.q_lora_rank,
            self.num_heads * self.q_head_dim,
            # 最终还需要做切分（split），一部分是 nope，一部分需要应用 rope
            bias=False,
        )

        # 同理对于 kv 也是一样的
        self.kv_down_proj = nn.Linear(
            self.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=config.attention_bias,
        )
        self.kv_down_layernorm = DeepseekV2RMSNorm(self.kv_lora_rank)
        self.kv_up_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads
            * (
                self.q_head_dim - self.qk_rope_head_dim + self.v_head_dim
            ),  # 其中 self.q_head_dim - self.qk_rope_head_dim 是 nope 部分
            bias=False,
        )

        # 对应公式 47 行
        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=config.attention_bias,
        )

        # 初始化 rope 的参数
        self.rotary_emb = DeepseekV2RotaryEmbedding(
            self.qk_rope_head_dim,
            self.max_postion_embeddings,
            self.rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        compressed_kv: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        MLA (Multi-head Linearized Attention) forward pass
        """
        bsz, q_len, _ = hidden_states.size()

        # 1. Query projection and split
        q = self.q_up_proj(self.q_down_layernorm(self.q_down_proj(hidden_states)))
        q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        q_nope, q_pe = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )

        # 2. Key/Value projection and split
        kv_seq_len = compressed_kv.size(1)
        compressed_kv, k_pe = torch.split(
            compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )

        # 备注：这里是的 ke_pe 长度和原来不一样了，用的不是 seq_len, 而是 kv_seq_len
        k_pe = k_pe.view(bsz, kv_seq_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        kv_up_proj = self.kv_up_proj.weight.view(self.num_heads, -1, self.kv_lora_rank)
        q_absorb = kv_up_proj[:, : self.qk_nope_head_dim, :]
        out_absorb = kv_up_proj[:, self.qk_nope_head_dim :, :]

        # 3. Apply RoPE to position-dependent parts
        print("q_pe shape:", q_pe.shape)

        cos, sin = self.rotary_emb(q_pe)
        q_pe = apply_rotary_pos_emb_v2(q_pe, cos, sin, position_ids)
        print("k_pe shape:", k_pe.shape)
        print("k pe mT shape:", k_pe.mT.shape)
        print("compressed_kv shape:", compressed_kv.shape)
        print("q_nope shape:", q_nope.shape)
        print("torch.matmul(q_pe, k_pe.mT) shape", torch.matmul(q_pe, k_pe.mT).shape)
        q_nope = torch.matmul(q_nope, q_absorb)
        attn_weights = (
            torch.matmul(q_pe, k_pe.mT)
            + torch.matmul(q_nope, compressed_kv.unsqueeze(-3).mT)
        ) / math.sqrt(self.q_head_dim)
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(q_nope.dtype)
        # shape is : (bsz, self.num_heads, q_len, kv_seq_len)

        # 2. Compute attention output
        attn_output = torch.einsum("bhql,blc->bhqc", attn_weights, compressed_kv)
        attn_output = torch.matmul(
            attn_output, out_absorb.mT
        )  # # torch.einsum('bhqc,hdc->bhqd', attn_output, out_absorb)
        attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights


# 写一个测试函数
def test_mlav2():
    config = DeepseekConfig(
        hidden_size=7168,
        num_heads=16,
        max_position_embeddings=1024,
        rope_theta=128000,
        attention_dropout=0.1,
        q_lora_rank=1536,
        qk_rope_head_dim=64,
        kv_lora_rank=512,
        v_head_dim=128,
        qk_nope_head_dim=128,
        attention_bias=False,
    )
    # 测试 MLAv2 attention
    # 生成 compressed_kv 的步骤:
    # 1. 先生成原始的 kv hidden states, shape 是 [bsz, kv_seq_len, hidden_size]
    # 2. 用 kv_a_proj_with_mqa 投影到 [bsz, kv_seq_len, kv_lora_rank + qk_rope_head_dim]
    # 3. split 成 compressed_kv 和 k_pe 两部分
    # 4. compressed_kv 过 layernorm
    # 5. k_pe 过 RoPE
    # 6. 最后把 compressed_kv 和 k_pe concat 在一起

    bsz = 2
    q_len = 1
    kv_seq_len = 12
    hidden_size = config.hidden_size

    # 生成测试数据
    q = torch.randn(bsz, q_len, hidden_size).cuda()
    # position_ids = torch.arange(q_len).expand(bsz, -1).cuda()
    # 注意这里和第一次的区别：这里只有最后一个 Q 的  token
    position_ids = torch.full((bsz, q_len), 12, dtype=torch.long).cuda()

    # 初始化模型
    model = MLAV2(config).cuda()

    # 先随机初始化一个 compressed_kv
    compressed_kv = torch.randn(
        bsz, kv_seq_len, config.kv_lora_rank + config.qk_rope_head_dim
    ).cuda()
    print(
        "compressed_kv shape:", compressed_kv.shape
    )  # [bsz, kv_seq_len, kv_lora_rank + qk_rope_head_dim]

    # 前向计算
    output, attn_weights = model(q, None, position_ids, compressed_kv)
    print("output shape:", output.shape)  # [bsz, q_len, hidden_size]
    print(
        "attn_weights shape:", attn_weights.shape
    )  # [bsz, num_heads, q_len, kv_seq_len]


test_mlav2()
```

### FAQ
Q: 为什么明明有矩阵吸收，在 forward 实现中，还是进行了两次乘法计算？
A: 从实际的测算中，对模型参数进行预处理，实际上耗时更久，具体测试见：[link](https://github.com/madsys-dev/deepseekv2-profile/blob/main/workspace/blog/optimizing-mla.md)


## ref 
- [https://zhuanlan.zhihu.com/p/700214123](https://zhuanlan.zhihu.com/p/700214123)
- [https://github.com/madsys-dev/deepseekv2-profile/blob/main/workspace/blog/optimizing-mla.md](https://github.com/madsys-dev/deepseekv2-profile/blob/main/workspace/blog/optimizing-mla.md)
- [https://mp.weixin.qq.com/s/E7NwwMYw14FRT6OKzuVXFA](https://mp.weixin.qq.com/s/E7NwwMYw14FRT6OKzuVXFA)
- [https://kexue.fm/archives/10091](https://kexue.fm/archives/10091)
- [https://www.armcvai.cn/2025-02-10/mla-code.html](https://www.armcvai.cn/2025-02-10/mla-code.html)
- 爱因斯坦方程的用法: https://zhuanlan.zhihu.com/p/71639781
- 假设没有矩阵吸收，可以看我的 blog: [从代码角度学习和彻底理解 DeepSeek MLA 算法](https://bruceyuan.com/post/hands-on-deepseek-mla.html)





## 其他
最后欢迎关注我，基本全网同名 [chaofa用代码打点酱油](https://bruceyuan.com/)
- 公众号（主要是为了订阅通知，不然看 Blog 就够了）： ![chaofa用代码打点酱油](https://bruceyuan.com/llms-zero-to-hero/chaofa-wechat-official-account.png)
- [B站-chaofa用代码打点酱油](https://space.bilibili.com/12420432)
- [YouTube-chaofa用代码打点酱油](https://www.youtube.com/@bbruceyuan)
- [chaofa 的 notion 简介](https://chaofa.notion.site/11a569b3ecce49b2826d679f5e2fdb54)