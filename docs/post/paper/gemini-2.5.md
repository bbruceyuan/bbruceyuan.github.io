---
title: "Gemini 2.5 Pro 是怎么炼成的？-- gemini 2.5 技术报告阅读笔记与思考"
date: 2025-07-13 19:20:20
tag:
  - LLM
  - paper
category:
  - paper-reading
description: "深入解读 Gemini 2.5 技术报告，分析多模态、长上下文与思考能力等核心突破，结合个人理解与行业趋势，快速掌握最新大模型技术发展。"
publish: true
permalink: /post/gemini-2.5-tech-report-reading-note.html
---

## 1. 收获（takeaway）

Gemini 的技术报告看上去比其他家的都小气一些，透露的细节非常的少，但是从行文来看，Gemini 2.5 Pro 成功的点主要有三个
-  多模态，其他的模型多模态能力或多或少都有所欠缺，只有 Gemini 2.5 这种模型才能有长视频的理解能力。
- LongContext，我理解可能是在强劲的基础架构下大力出奇迹的结果。
- 思考能力，Thinking is All you Need. This is very suitable for Agent.

## 2. 模型结构、训练和数据

### 2.1 模型结构（Model Architecture）与长下文能力

2.5 系列的模型是基于 MoE 的 Transformers 模型，具体做法是：每一个 token 都会动态路由到某一个 Expert 中，这种方法叫做 Sparse MoE，具体实现可以参考；[LLM MOE的进化之路，从普通简化 MOE，到 sparse moe，再到 deepseek 使用的 share_expert sparse moe](https://yuanchaofa.com/llms-zero-to-hero/the-way-of-moe-model-evolution.html)。

文中提到模型架构的修改对于 Gemini 2.5 效果的提升具有重要的作用，加上 Google 在预训练稳定性、动态优化等技术让最终效果比上一代有巨大的提升。

此外，关于模型的长下文能力（long-context）并没有提到是怎么训练的（比如RoPE 做位置编码，用 YaRN，NTK 等方法做上下文扩展)，但是有一个非常重要的细节，**训练数据的上下文长度就已经达到了 1M**，也就是说有可能 Google 的 LongContext 能力就是硬训练出来的。这里说一下 chaofa 个人的猜测，这里可能用到了很多 Long-Video，Long-Audio 等多模型相关的数据来提升模型的 Long-Context 能力。

下图是一个成本曲线和能力曲线，在相同成本下 Gemini2.5 Pro 效果很好，另外非常出彩的一点是，Gemini 2.5 flash 成本非常的便宜。

![gemini-tech-report-readding-notes-1752401580682.webp](https://cfcdn.bruceyuan.com/blog/2025/gemini-tech-report-readding-notes-1752401580682.webp)


> gemini 2.5 开发去重验证集的污染主要通过 n-gram 方式，辅助用语义相似性和模型过滤验证集数据。

### 2.2 预训练（Pre-Training）

预训练数据集是一个大规模、多样化的数据集合，涵盖广泛的领域和模态，包括公开的网页文档、代码（多种编程语言）、图像、音频（包括语音和其他类型音频）以及视频，其中2.0版本的数据截止日期为2024年6月，2.5版本为2025年1月。和上一代相比，提升了「质量过滤、去重」的能力。

> 除了多模态这一点，说实话等于什么也没说🤔

### 2.3 后训练（Post-Training）

PostTraining 训练数据包含包含多模态数据的集合，配有成对的指令和响应，此外还包括人类偏好和工具使用数据。所以这是现在大模型天然就是一个 Agentic Model。

自Gemini 1.5 首次发布以来，在监督微调（SFT）、奖励建模（RM）和强化学习（RL）各个阶段持续关注数据质量的推动下，后训练方法方面取得了重大进展。**一个关键的重点是利用模型本身协助这些过程，从而实现更高效且细致的质量控制**（重点的事情说三遍，自己原有模型做质量过滤、合成等）。

此外，还增加了用于强化学习的训练算力，使得对模型行为的深入探索和优化成为可能。这一改进结合了**可验证奖励和[基于模型的生成奖励](https://yuanchaofa.com/post/deepseek-grm-paper-reading-notes.html)** 的应用，提供了更复杂且可扩展的反馈信号。同时，对强化学习流程的算法调整也提升了长时间训练的稳定性。这些进步使 Gemini 2.5 能够从更加多样和复杂的强化学习环境中进行学习，包括那些需要**多步骤操作和工具使用的环境**。

### 2.4 模型思考能力

![gemini-tech-report-readding-notes-1752402585776.webp](https://cfcdn.bruceyuan.com/blog/2025/gemini-tech-report-readding-notes-1752402585776.webp)

从上图可以先得出一个结论，有 Thinking 比没有更好，2.5 的[动态 Thinking](https://yuanchaofa.com/post/slow-fast-thinking-from-qwen3-thinking-mixed-to-adacot-to-adathinking.html) 效果更好，这里的动态思考的提升不仅仅是来自于动态思考本身，还有数据质量的提升，以及更多领域上的思考训练，而不仅仅是数学代码等领域。

思维能力可以和 Gemini 其他功能进行整合，包括原生多模态输入（图像、文本、视频、音频）和 Long-Context（超过100万个token）。对于这些功能中的任何一个，模型会自行决定在给出答案之前需要思考多长时间。我们还提供了设置思维预算的能力，限制模型在指定的token数量内作出回应。如下图所示，增加这一预算可以使模型提升其性能，并显著提高准确率。

![gemini-tech-report-readding-notes-1752402901438.webp](https://cfcdn.bruceyuan.com/blog/2025/gemini-tech-report-readding-notes-1752402901438.webp)

### 2.5 其他特定能力的提升

从论文的描述看，这里的提升原来于两个方面；
- 配备合适的数据和工具
- 提供更有指导的优化目标（类似于 Shunyu Yao 提出的 [The Second Half](https://ysymyth.github.io/The-Second-Half/)）

#### 代码（Code）

可能是看到 Claude 成功，Google 战略性的改变，也就是需要提升模型 Code 能力。自 Gemini 1.5 以来，gemini 在预训练和后续训练阶段都进行了协同努力。

在预训练方面，加强了从代码仓库和网络来源中纳入**更大数量和更多样化代码数据**的力度，**「从而迅速扩展了覆盖范围」**，并推动了更高效计算模型的发展。此外，gemini 还大幅增强了用于评估与下游应用场景相匹配代码能力的评测指标体系，同时提升了准确预测模型性能的能力。

在后续训练阶段，开发了融合推理能力的全新训练技术，并精心挑选了一组多样化的工程任务，旨在为 Gemini 配备解决现代工程挑战所必需的高效问题解决技能。展现这些进步的关键应用包括 IDE 功能、针对完整代码仓库中复杂多步骤操作的代码代理用例，以及端到端网页和移动应用开发等多模态交互场景。

> 核心观点是：更贴近实际开发的训练数据和目标。

#### 事实性（Factuality）

增强模型的世界知识及其根据提示中提供的上下文内容进行忠实回答的能力，为了满足这些新需求，Gemini 2.0 实现了重大突破，成为我们首个原生集成工具调用能力的模型家族，例如可直接调用Google Search，从而能够构造精准的查询语句，并结合最新信息与来源进行综合回答。在此基础上，Gemini 2.5进一步融合了高级推理能力，使其能够在内部思维过程与搜索功能之间灵活切换，以应对复杂的多跳查询并执行长期任务。该模型已掌握使用搜索及其他工具的方法，能对工具输出进行推理，并提出进一步、详细的后续查询，以扩展可用信息并验证回答的事实准确性。

> 内在的 Search Tool 调用能力，能够切换内在思考与搜索工具。


#### 长上下文 (LongContext)
建模和数据方面的进展帮助我们提升了模型在使用百万级上下文窗口时对查询的响应质量，同时我们也重新设计了内部评估体系，使其更具挑战性，以更好地指导建模研究方向。在优化过程中，我们将目标设定为具有挑战性的检索任务（如LOFT，Lee等人，2024）、长上下文推理任务（如MRCR-V2，Vodrahalli等人，2024）以及多模态任务（如VideoMME，Fu等人，2025）。根据表6中的结果，新的2.5版本模型相比之前的Gemini 1.5模型实现了显著提升，并在所有这些任务上达到了业界领先水平。其中一个展示视频回忆能力提升的示例可以体现这些进步。

#### 多语言能力（Multilinguality）

对预训练和微调数据质量的精细优化、分词技术的提升、核心建模方法的创新以及有针对性的能力迭代优化。改进效果在印度语系以及中日韩语系中尤为显著，这些语言通过数据质量和评估方面的专项优化，实现了质量和解码速度的大幅提升。因此，用户能够享受到更出色的语言适配性，即输出内容更加忠实于所请求的目标语言，同时在多种语言中的生成质量和事实准确性也得到了有力增强，进一步巩固了 Gemini 在多样语言环境下的可靠性。
#### 音频（Audio）

Gemini 1.5 主要专注于原生音频理解任务，如转录、翻译、摘要和问答，而在 Gemini 2.5 中，模型进一步具备了音频生成能力，例如文本到语音合成，以及从原生音视频中生成对话音频。为了实现低延迟的流式对话，我们引入了因果音频表示方法，使得音频可以以流式方式输入和输出 Gemini 2.5。这些能力得益于覆盖超过 200 种语言的大量预训练数据，以及更优化的后训练方法。最终，通过改进后的后训练流程，我们将思考能力、情感对话、情境感知和工具使用等高级功能集成到了 Gemini 的原生音频模型中。

#### 视频（Video）

大幅扩展了预训练和后训练阶段的视频理解数据，从而提升了模型对音视频内容及时间维度的理解能力。同时优化了模型训练，使其每帧仅需使用 66 个视觉标记，而非原有的 258 个，使得在1M标记上下文窗口中可处理约3小时的视频内容，而不是原来的1小时。这些改进催生了两项此前无法实现的新应用：一是从视频中创建交互式应用程序（例如用于测试学生对视频内容理解的小测验），二是生成p5.js动画以展示视频中的关键概念。

#### Agent（Deep Research）

Gemini Deep Research是一个基于Gemini 2.5 Pro模型构建的智能体，旨在战略性地浏览网络，为即使是最细分的用户查询提供有依据的回答。该智能体经过优化，能够进行任务优先级排序，并能在浏览过程中识别何时已进入死胡同。

## 3. 其他

最后欢迎来探讨对世界的认知，基本全网同名 [chaofa用代码打点酱油](https://yuanchaofa.com/) (推荐)
- [公众号-chaofa用代码打点酱油](https://yuanchaofa.com/llms-zero-to-hero/chaofa-wechat-official-account.png)
	- ![chaofa用代码打点酱油](https://yuanchaofa.com/llms-zero-to-hero/chaofa-wechat-official-account.png)
- [B站-chaofa用代码打点酱油](https://space.bilibili.com/12420432)
- [YouTube-chaofa用代码打点酱油](https://www.youtube.com/@bbruceyuan)
- [chaofa 的 notion 简介](https://chaofa.notion.site/11a569b3ecce49b2826d679f5e2fdb54)
- [X(Twitter)-chaofa用代码打点酱油](https://x.com/bbruceyuan)

