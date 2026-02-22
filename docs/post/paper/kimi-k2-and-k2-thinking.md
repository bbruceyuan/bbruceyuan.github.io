---
title: Kimi-K2 和 Kimi-K2-Thinking 深度解读：从预训练优化到 Agentic 能力训练的完整流程（含MuonClip优化、Agentic 数据合成等）
date: 2025-11-09 17:26:00
tag:
  - LLM
  - paper
category:
  - paper-reading
description: 深度解读 Kimi K2 和 K2 Thinking 技术细节：MuonClip 优化方案、大规模 Agentic 数据合成 pipeline、通用强化学习的 Self-Judging 机制，以及 200-300 步工具调用的 Test-Time Scaling。从预训练到后训练，揭秘月之暗面如何打造 SOTA 开源 Thinking 模型。
publish: true
permalink: /post/kimi-k2-and-kimi-k2-thinking-notes.html
---


## 0. 背景

月之暗面发布的 **[Kimi K2 Thinking](https://moonshotai.github.io/Kimi-K2/thinking.html)**，在 Humanity's Last Exam (HLE) 上达到了 44.9% 的成绩，在多个基准测试中表现优异，不过榜单简单看一眼即可；让我比较惊喜的是，K2 Thinking 可以执行 200-300 步连续的工具调用，有类似于 `claude` 一样的长程规划和自适应推理能力。

但是，K2 Thinking 的官方 blog 只展示了 benchmark 数据和 demo，并没有透露具体的技术细节。作为一个大模型从业者，看到 Twitter/知乎大家都在聊这个模型，所以我就比较好奇「模型的训练方法」以及「给我们工作学习中的启发」。

好在今年早些时候发布了 **Kimi K2** 的完整[技术报告](https://arxiv.org/abs/2507.20534)和 [技术 blog](https://moonshotai.github.io/Kimi-K2/)。而 **K2 Thinking 和 K2 师出同源**，只是在 K2 的基础上增加了 thinking 能力，更强的工具调用能力，通过 test-time scaling 实现一个更强的 Thinking Agent。因此，通过深入研究 K2 的技术细节，我们就能理解 K2 Thinking 是如何炼成的。

我是朝发（CHAOFA）这篇文章会从 K2 的技术报告出发，结合 K2 Thinking 的特点，了解这个 SOTA 开源 thinking 模型是怎么训出来的。**核心关注三个问题**：

1. **预训练阶段**：如何用 MuonClip 优化器实现更高的 token 效率？
2. **后训练阶段**：如何通过大规模 Agentic 数据合成和通用强化学习，让模型学会使用工具？
3. **Test-Time Scaling**：如何让模型在推理时进行长程思考和工具调用？

> 历史上此比较相关文章：
>
> - [深度解读 Kimi-K1.5，真正了解 RL 数据是怎么筛选的](https://yuanchaofa.com/post/kimi-k1.5-paper-reading-notes.html)
> - [自顶向下方式深度解读 DeepSeek-R1，内含大量细节](https://yuanchaofa.com/post/deepseek-r1-paper-reading-notes.html)
> - [自适应快慢思考推理模型（Adaptive Reasoning Model）：Qwen3混合思考->字节AdaCoT->清华AdaptThinking](https://yuanchaofa.com/post/slow-fast-thinking-from-qwen3-thinking-mixed-to-adacot-to-adathinking.html)

> 如果不喜欢看文字可以看视频解读，[B 站-chaofa用代码打点酱油](https://www.bilibili.com/video/BV1yikRBvEwy/)和 [YouTube](https://www.youtube.com/@bbruceyuan)
>
> [算法视角深度解读 Kimi K2 和 K2 Thinking，从预训练优化到 Agentic 能力训练的完整流程（含MuonClip优化、Agentic 数据 --bilibili](https://www.bilibili.com/video/BV1yikRBvEwy/)

## 1. 整体架构：从 K2 到 K2 Thinking

![image.png|700x366](https://cfcdn.yuanchaofa.com/blog/2025/20251109144342.png)

> Archiecture from: [Sebastian Raschka](https://x.com/rasbt/status/1986511951141441648?s=20)

先来看一下上面的整体结构图，然后在深入技术细节之前，我们有必要先理解 K2 和 K2 Thinking 的关系。

### 1.1 K2：Open Agentic Intelligence 的基座

Kimi K2 是一个 [MoE (Mixture-of-Experts)](https://yuanchaofa.com/llms-zero-to-hero/the-way-of-moe-model-evolution.html) 模型，拥有 **32B 激活参数和 1T 总参数**。它在非 thinking 模型中，在前沿知识、数学和编码任务上达到了 SOTA 性能。

K2 的核心特点是**有比较强的 Agentic 能力**。什么是 Agentic 任务？就是模型不仅要回答问题，还要主动使用工具、执行操作、完成复杂的多步骤任务。比如：

- 用 Python 分析数据、生成可视化网页
- 在命令行中编辑文件、运行命令
- 通过搜索和浏览器收集信息、验证假设、构建答案

K2 发布了两个版本：

- Kimi-K2-Base：基础模型，适合研究者/开发者/企业用户进行微调
- Kimi-K2-Instruct：后训练模型，适合直接使用，是一个非推理模式（Non-Reasoning Model）

### 1.2 K2 Thinking：加入 Test-Time Scaling

Kimi K2 Thinking 是在 K2 的基础上，通过额外的训练，让模型具备了 thinking 能力。它的核心特点是：

1. 边思考边使用工具：模型在推理过程中，会进行 `think → search → browse → think → code` 的循环，动态生成和验证假设
2. 长程推理：可以执行 200-300 步连续的工具调用，保持推理的连贯性。（这点是让人比较惊喜的）
3. Test-Time Scaling：通过增加推理时的 thinking tokens 和工具调用步数，提升模型性能

从架构上看，`K2 Thinking = K2 + Thinking Ability + Test-Time Scaling`。因此，**理解 K2 的训练方法，就能理解 K2 Thinking 的 80%**。

下面我们按照训练流程，依次讲解预训练、后训练和 test-time scaling 的关键技术。

## 2. 预训练

### 2.1 基于 MuonClip 优化器的 Token 效率优化

预训练是 Agentic Intelligence 的关键基础，它建立了让强化学习探索变得可行、高效和可泛化的先验知识。但是，正如 Ilya Sutskever 所说，数据是有限的"化石燃料"，其增长速度远远落后于算力的增长。这使得**预训练阶段的 token 利用效率**成为 AI scaling laws 中的新关键系数。

#### 2.1.1 为什么需要更好的优化器？

给定一个大致有限的预训练数据集和固定的模型配置，更 token 高效的优化器能产生更多的智能。Moonshot 之前的工作 [Moonlight](https://github.com/MoonshotAI/Moonlight) 已经证明，[Muon](https://kellerjordan.github.io/posts/muon/) 优化器在 LLM 训练中显著优于广泛使用的 AdamW 优化器，即“相同配置训练下有更低的 loss”。

K2 的设计目标是进一步扩展 Moonlight，它采用了类似 DeepSeek-V3 的架构。基于 scaling-law 分析，他们做了两点改进（看图更清晰）：

- 减少了 attention heads 的数量，以提高长上下文效率。
- 增加了 MoE 的稀疏性，以获得更高的 token 效率

> 原文这么写的：Based on scaling-law analysis, we reduce the number of heads for long-context efficiency, and increase MoE sparsity for greater token efficiency。

但在扩展过程中，他们遇到了一个持续的挑战：**由 attention logits 爆炸引起的训练不稳定**。这个问题在使用 Muon 时更频繁，而在 AdamW 中较少。现有的解决方案（如 Qwen3 用的 query-key normalization）都不够充分（防止数值溢出）。

#### 2.1.2 MuonClip：直接控制 Attention Logits

为了解决这个问题，kimi 提出了 MuonClip 优化器，它通过 **qk-clip 技术**改进了 Muon。

**核心思想**：qk-clip 通过在 Muon 更新后**直接重新缩放 query 和 key 投影的权重矩阵**，从源头控制 attention logits 的规模，从而稳定训练。（注意：这里是更新完之后，所以不会改变这一次更新的 forward/backward 操作，影响的是下一步）。

具体来说，query 和 key 投影按如下方式缩放：

$$
q_i = \eta^{\alpha} W_q x_i
$$

$$
k_i = \eta^{1-\alpha} W_k x_i
$$

其中 $\alpha$ 是一个平衡超参数，因此 attention logit 变为：

$$
(\eta^{\alpha} q_i)^\top (\eta^{1-\alpha} k_j) = \eta\, q_i^\top k_j
$$

自适应因子 $\eta$（阈值为 $t$）在每一步之后根据该步的最大 attention logit 设置：

$$
\eta = \min\left(\frac{t}{\displaystyle\max_{i,j}\bigl(q_i^\top k_j\bigr)}, 1\right)
$$

其中 $t$ 是预设的阈值。这是一个通用技术，可能适用于其他稳定化场景。这里其实还有一些其他的细节，比如 每个 head 有不同的 $\eta$。

#### 2.1.3 实验结果：零训练尖峰

实验表明，MuonClip 有效地防止了 logit 爆炸，同时保持了下游任务性能。在实践中，K2 使用 MuonClip 在 15.5T tokens 上进行预训练，实现了零训练尖峰（zero loss spike），证明了 MuonClip 是大规模 LLM 训练的稳健解决方案。

![image.png|700x420](https://cfcdn.yuanchaofa.com/blog/2025/20251109161151.png)

从 loss 曲线可以看出，MuonClip 的训练过程非常平滑，没有出现任何不稳定的情况。这为后续的 Agentic 能力训练打下了坚实的基础。

**小结**：MuonClip 优化器通过 qk-clip 技术，在保持 Muon 高 token 效率的同时，解决了训练不稳定问题，使得在同等条件下获得比 AdamW 更低的 loss，使得 K2 能够在有限的数据上训练出更强的基础模型。

### 2.2 文本的改写优化

K2 相比 K1.5 的一个关键进步是引入了**合成数据生成策略**来提高 token 利用率。核心思想是：通过精心设计的改写 pipeline，在不引入显著过拟合的情况下，扩大高质量 tokens 的数量。改写（Rephrasing） 就是数据合成的一种方式，主要是为了提高「高质量数据的占比」，尤其是「知识领域」和「数学领域」。：

#### 2.2.1 知识领域数据改写

在知识密集型文本上进行预训练面临一个权衡：单次 epoch 不足以全面吸收知识，而多次 epoch 重复会导致收益递减并增加过拟合风险。为了提高高质量知识 tokens 的利用率，K2 提出了一个合成改写框架，每个语料库最多改写两次，包含三个关键组件：

**A. 风格和视角多样化的提示（Style- and perspective-diverse prompting）**

通过精心设计的 prompts，引导大语言模型以不同的风格和视角生成原文的忠实改写。这样做的好处是：

- 增强语言多样性
- 保持事实完整性
- 避免简单的同义词替换

**B. 分块自回归生成（Chunk-wise autoregressive generation）**

为了在长文档中保持全局连贯性并避免信息丢失，采用基于分块的自回归改写策略，一图胜千言：

![image.png|700x344](https://cfcdn.yuanchaofa.com/blog/2025/20251109163028.png)

**C. 保真度验证（Fidelity verification）**

为了确保原文和改写内容之间的一致性，进行保真度检查，比较每个改写段落与其源文本的语义对齐。这是训练前的初步质量控制步骤。

#### 2.2.2 数学领域数据改写

为了增强数学推理能力，K2 采用了两种策略：

**A. "学习笔记"风格改写**

将高质量的数学文档改写成"学习笔记"风格，遵循 [SwallowMath](https://arxiv.org/abs/2501.01926) 中引入的方法。这种风格更接近人类学习数学的方式，包含：

- 逐步推导过程
- 关键概念解释
- 示例和练习

**B. 多语言翻译**

将其他语言的高质量数学材料翻译成英语，以增加数据多样性。这样可以：

- 利用非英语世界的优质数学资源
- 增加数学表达的多样性
- 扩大训练数据规模

**小结**：通过针对知识和数学领域的专门改写技术，K2 在不显著增加过拟合风险的情况下，大幅提高了高质量 tokens 的利用率。这种受控的数据增强策略是 K2 预训练成功的关键因素之一。

## 3. 后训练(重点)

K2 的增强 Agentic 能力源于两个重要方面：

- **大规模 Agentic 数据合成**
- **通用强化学习**

### 3.1 大规模 Agentic 数据合成：教会模型使用工具

为了教会模型复杂的工具使用能力，kimi 是基于**大规模模拟真实世界的工具使用场景**，构建了数据 pipeline。

#### 3.1.1 数据合成流程

这个管道的核心思想是：**系统地演化数百个领域，包含数千个工具**（包括真实的 MCP 工具和合成工具），然后生成数百个具有不同工具集的 agents。

![image.png|700x232](https://cfcdn.yuanchaofa.com/blog/2025/20251109175053.png)

辅助看这个图：

![image.png|700x233](https://cfcdn.yuanchaofa.com/blog/2025/20251109163203.png)

具体流程如下：

1. 定义领域和工具：涵盖各种真实场景，如数据分析、网页开发、系统管理等
2. 生成任务：所有任务都是基于 rubric 的（有明确的评分标准），确保一致的评估
3. 模拟交互：Agents 与模拟环境和用户 agents 交互，创建真实的多轮工具使用场景
4. LLM 评判(LLM as judge)：根据任务 rubrics 评估模拟结果，过滤出高质量的训练数据

这个可扩展的 pipeline 生成了**多样化、高质量的数据**，为大规模拒绝采样和强化学习铺平了道路。

#### 3.1.2 为什么这个方法有效？

传统的工具使用训练依赖于人工标注的数据，成本高、规模小、多样性有限。而 k2 的方法通过**自动化合成**，可以：

- 无限扩展：只要定义新的领域和工具，就能生成新的训练数据
- 保证质量：通过 rubric-based 评估和 LLM judge，确保数据质量
- 覆盖长尾场景：可以模拟各种罕见但重要的工具使用场景

### 3.2 通用强化学习：不可验证奖励

传统的强化学习主要应用于**可验证奖励**的任务，比如数学题（答案对错明确）和竞赛编程（能否通过测试用例）。但对于**不可验证奖励**的任务（如写研究报告、创意写作），传统 RL 就无能为力了。

> 是不是突然想起了 [DeepSeek-GRM（通用奖励模型）](https://yuanchaofa.com/post/deepseek-grm-paper-reading-notes.html)。

#### 3.2.1 Self-Judging 机制

核心思想是：**模型作为自己的评判者**，为不可验证的任务提供可扩展的、基于 rubric 的反馈。

具体做法：

1. 对于不可验证的任务，模型生成多个候选答案
2. 模型自己根据 rubric 评估这些答案，给出分数
3. 使用这些分数作为奖励信号，进行强化学习

> 但这里有个问题：模型的自我评估准确吗？这不还是 LLM as Judge 那一套吗？

#### 3.2.2 用可验证奖励改进 Critic

kimi 的解决方案是：**在可验证奖励的 on-policy rollouts 中，持续更新 critic**，使 critic 在最新策略上不断提高评估准确性。

这可以看作是**用可验证奖励来改进不可验证奖励的估计**。通过这种方式，模型的自我评估能力会随着训练不断提升，从而支持更广泛的任务。

**小结**：通过大规模 Agentic 数据合成和通用强化学习，K2 学会了在各种场景下使用工具，并且能够处理可验证和不可验证的任务。这为 K2 Thinking 的长程推理能力打下了基础。

## 4. K2 Thinking

K2 Thinking 在 K2 的基础上，增加了 **thinking 能力**、**更强的工具调用能力**和 **test-time scaling**。这使得模型能够在推理时进行长程思考和工具调用，从而解决更复杂的问题。

### 4.1 什么是 Test-Time Scaling？

Test-Time Scaling 是指**在推理时增加计算量，以提升模型性能**。对于 K2 Thinking，这体现在两个方面：

1. 增加 thinking tokens：模型在生成答案前，会先生成大量的思考过程（类似 OpenAI o1，这其实就是 Long-CoT，这种技术在 Kimi-k1.5 就已经开始做了）
2. 增加工具调用步数：模型可以执行 200-300 步连续的工具调用，进行长程规划（这是新增的，为了 Agentic 能力的提升）

这两者结合，使得 K2 Thinking 能够解决需要深度推理和多步操作的复杂问题。

### 4.2 边思考边使用工具：Interleaved Reasoning

K2 Thinking 的核心能力是**边思考边使用工具**。它会进行动态的 `think → search → browse → think → code` 循环，这个循环可以重复数百次，直到找到答案：

1. Think：分析问题，生成假设
2. Search：搜索相关信息
3. Browse：浏览网页，提取关键信息
4. Think：验证假设，调整策略
5. Code：编写代码，执行计算

### 4.3 简要看看 benchmark

#### 4.3.1 Agentic Search：超越人类基线

在 BrowseComp benchmark 上，K2 Thinking 达到了 **60.2%** 的成绩，显著超越了 **29.2%** 的人类基线。

BrowseComp 是一个挺具有挑战性的 benchmark，旨在评估模型**持续浏览、搜索和推理难以找到的真实世界网络信息**的能力。

#### 4.3.2 Agentic Coding：构建完整的应用

K2 Thinking 在编码任务上也表现出色：

![image.png|700x469](https://cfcdn.yuanchaofa.com/blog/2025/20251109164955.png)

从官网看，K2 Thinking 可以**从单个 prompt 构建完整的应用**，包括：

- 组件密集的网站
- Word 克隆应用
- 交互式数据分析工具

### 4.4 小结

通过 test-time scaling，K2 Thinking 能够在推理时进行长程思考和工具调用，从而解决需要深度推理和多步操作的复杂问题。这使得它在 Agentic Reasoning、Agentic Search 和 Agentic Coding 任务上都达到了 SOTA 性能。（有点 claude 那味道了）

## 5. 技术细节对比：K2 vs K2 Thinking

让我们总结一下 K2 和 K2 Thinking 的关键区别：

| 维度 | K2 (Instruct) | K2 Thinking |
|------|---------------|-------------|
| 模型类型 | Non-thinking（无长思考） | Thinking model（有长思考） |
| 推理方式 | 直接生成答案 | 边思考边使用工具 |
| 工具调用 | 支持，但步数有限（其实也挺好的） | 200-300 步连续调用 |
| Test-Time Scaling | 不支持 | 支持（thinking tokens + 工具调用） |
| 适用场景 | 通用对话、快速响应 | 复杂推理、长程规划 |

从技术角度看，K2 Thinking 是在 K2 的基础上：

- 增加了 thinking 能力：通过额外的训练，让模型学会生成长思考过程
- 优化了工具调用：支持更长的工具调用链，保持推理连贯性
- 引入了 test-time scaling：在推理时增加计算量，提升性能

## 6. 核心启发：我们能学到什么？

从 K2 和 K2 Thinking 的技术报告中，我觉得比较重要的，可能能在我们实际业务中用上的点有：

1. 数据的改写策略，写的很详细，尤其是在做「创意写作」方面工作的同学。
2. Agentic 训练数据构建的 pipeline。别扯没用的，就是要「真实环境模拟运行」获取大量的 trajectory 然后用 LLM 做筛选。（以前我只想着去构造，迭代去筛选更有效）
3. rubric-based 评估（不过这个其实一两年前大家就在用了，为什么突然又改头换面火了一下，这个真的太考验业务敏感度和怎么使用了，能直接在 k2 这种级别的开源模型上搞出来，还是挺佩服的）
4. test-time scaling 还是很有必要的，梦回年初 Long-CoT，想要效果好牺牲点时间绝对是值得的。（尽管可能会导致过度生成、倾向于用工具的问题）

> 个人碎碎念：
>
> 1. 从 K1.5 发 paper 开始，就感觉 KIMI 突然开始醒悟做社区了，OpenSource 真的是比较博好感，现在中国的开源模型真的牛皮🐂🍺
>
> 2. 好像 `claude` 在 coding 上的爆火让大家都领悟到了 `agentic` 能力的重要性。希望把 Claude 价格打下来，加油～

## 7. 更多内容

欢迎关注我，我是朝发（chaofa），基本全网同名 [chaofa用代码打点酱油](https://yuanchaofa.com/)

- 公众号：![chaofa用代码打点酱油](https://yuanchaofa.com/llms-zero-to-hero/chaofa-wechat-official-account.png)
- [B站-chaofa用代码打点酱油](https://space.bilibili.com/12420432)
- [YouTube-chaofa用代码打点酱油](https://www.youtube.com/@bbruceyuan)
- [chaofa 的 notion 简介](https://chaofa.notion.site/11a569b3ecce49b2826d679f5e2fdb54)
