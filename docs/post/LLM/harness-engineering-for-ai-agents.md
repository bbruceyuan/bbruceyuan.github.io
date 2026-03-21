---
title: "Harness Engineering — Agent 不好用，也许不是模型的问题"
date: 2026-03-14 22:00:00
tag:
  - agent
  - Agent Harness
  - Harness Engineering
  - Context Engineering
  - Coding Agent
category:
  - hands-on-code
description: "同一个模型，只改 Agent Harness，性能从 Top 30 到 Top 5。Harness Engineering 到底是什么？和 Context Engineering 什么关系？以及 The Bitter Lesson 再思考。"
publish: true
permalink: /post/harness-engineering-for-ai-agents.html
star: true
image: /diagrams/harness-engineering/harness-overview.svg
---

## 0. 阅读收获 (Takeaway)

读完本文，你将了解：

- **Harness 到底是什么**：一个具体的、可操作的定义，不是又一个抽象类比
- **Prompt → Context → Harness 的演进**：三者的关系、边界和社区争议
- **同一个模型，只改 Harness，性能从 Top 30 到 Top 5**：一个具体案例的拆解
- **哪些设计是持久的，哪些会被模型进步淘汰**：对 The Bitter Lesson 的再思考

> 前置知识：如果你对 Agent 的上下文管理还不太熟，建议先看 [Agent 系统中的 Prompt Caching 设计（上）](/post/prompt-cache-design-for-llm-agents) 和 [（下）](/post/agent-context-management-and-sub-agents)，本文会多次引用。

---

## 1. Agent 不好用？

用 Coding Agent 的人大概都有过这种经历：Agent 写了一坨代码，自己看了一眼觉得"嗯不错"，然后就停了。测试？没跑。边界情况？没想。你花了 20 分钟等它干活，最后还是得自己收拾烂摊子。

本能反应通常是："模型还是不够聪明，等 GPT-6 就好了。"

但 [HumanLayer](https://www.humanlayer.dev/blog/skill-issue-harness-engineering-for-coding-agents) 团队在做了上百个 Agent 项目后，得出了一个不同的结论：

> "It's not a model problem. It's a configuration problem."

LangChain 用数据证明了这一点：**同一个模型（GPT-5.2-Codex），只改模型之外的东西，Terminal Bench 2.0 得分从 52.8 涨到 66.5，排名从 Top 30 到 Top 5。**

模型之外的那些东西，现在有了一个统一的名字：**Harness**。


## 2. Harness 是什么？

LangChain 的 Vivek Trivedy 给了一个最干净的定义：

> **Agent = Model + Harness**. If you're not the model, you're the harness.

Harness 就是**除了模型权重以外的一切**。

![Harness Overview](/diagrams/harness-engineering/harness-overview.excalidraw)


这不是抽象类比。打开你的 _Claude Code_，它的 Harness 具体包括：

| 组件 | 具体是什么 | 解决什么问题 |
|------|-----------|-------------|
| System Prompt | 系统提示词，定义行为规范 | 模型不知道自己该扮演什么角色 |
| CLAUDE.md / AGENTS.md | 项目级指令文件，启动时注入 | 模型不知道这个项目的规范和上下文 |
| Tools (bash, 文件读写, 浏览器) | 可调用的工具集 | 模型只能输出文本，不能执行操作 |
| Sandbox | 隔离的执行环境 | 不能让模型直接操作你的生产环境 |
| Compaction | 上下文压缩机制 | Context Rot，长对话性能衰减 |
| Hooks / Middleware | 在模型调用前后插入的逻辑 | 模型不会主动自检、容易死循环 |
| Sub-agent 管理 | 子代理派生和结果回收 | 单个上下文窗口装不下复杂任务 |

如果你读过我之前写的 [Prompt Caching 系列](/post/prompt-cache-design-for-llm-agents)，会发现上下文管理、Compaction、子代理架构——这些其实都是 Harness 的组成部分。


[My AI Adoption Journey](https://mitchellh.com/writing/my-ai-adoption-journey) 有一个更偏工程实践的定义：

> "Harness engineering is the idea that anytime you find an agent makes a mistake, you take the time to engineer a solution such that the agent never makes that mistake again."

Agent 犯了错 → 不是改 prompt 重试 → 而是**改环境让它不可能再犯这个错**。这就是 Harness Engineering 的核心思想。


## 3. Prompt → Context → Harness：不是替代，是包含

这三个概念的关系经常被误解为"一代比一代强，后面的替代前面的"。但实际上它们是**包含关系**：


![Prompt Context Harness Layers](/diagrams/harness-engineering/prompt-context-harness-layers.excalidraw)

简单区分：

| | 管什么 | 解决什么 | 失败了怎么办 |
|---|-------|---------|-------------|
| **Prompt Engineering** | 一条指令 | 怎么问才能得到好答案 | 改 prompt 重试 |
| **Context Engineering** | 整个上下文窗口 | 怎么给够信息完成复杂任务 | 调整检索/压缩/隔离策略 |
| **Harness Engineering** | 模型之外的一切 | 怎么建系统让 Agent 可靠地长期工作 | 改环境让错误不可能再发生 |

- **Prompt Engineering** 大家已经很熟了——few-shot、CoT、角色扮演。
- **Context Engineering** 是 2025 年 Karpathy 推动的概念升级。核心思想是：不要只关注 prompt 本身，要关注**整个上下文窗口的管理**。LangChain 总结了四个策略：Write（写到外部存储）、Select（按需检索）、Compress（压缩）、Isolate（隔离子任务的上下文）。
- **Harness Engineering** 再往外扩一层：不只管上下文窗口内的事，还管**执行环境、工具、持久化、反馈循环、架构约束**这些上下文窗口之外的东西。

换句话说：Context Engineering 解决了"给模型什么信息"的问题，Harness Engineering 还要解决"给模型什么环境"的问题。

社区对这个关系其实有争议：LangChain 认为 Harness ⊃ Context（超集）；HumanLayer 认为 Harness ⊂ Context（子集）。

> **我的判断是超集关系更合理**——sandbox、CI gate、linter 这些东西显然不属于"上下文管理"，但它们是 Harness 的核心组成部分。不过争论定义意义不大，重要的是意识到：Agent 出问题的时候，除了看"给了什么信息"，还要看"跑在什么环境里"。


## 4. 案例：同一个模型，只改 Harness

LangChain 做过一个实验，很好地说明了 Harness 的实际影响。

基线：deepagents-cli + GPT-5.2-Codex，默认配置，Terminal Bench 2.0 得分 52.8。他们只调了三个"旋钮"（原话是 "knobs on a harness"）：System Prompt、Tools、Middleware。最终得分 66.5，排名从 Top 30 到 Top 5。

具体改了什么：

### 4.1 修改 1：退出前强制自检

最常见的失败模式：Agent 写完代码 → 重读一遍自己的代码 → 觉得"看起来没问题" → 停止。根本没跑测试。

他们加了一个 Middleware：在 Agent 试图退出时拦截，强制注入一个 checklist 让它对照任务说明验证。这本质上是一个 [Ralph Wiggum Loop](https://ghuntley.com/loop/)——hook 住退出，强制继续。

### 4.2 修改 2：启动时注入环境信息

Agent 在陌生环境中会浪费大量时间探索目录结构、找 Python 环境。他们在 Agent 启动时自动跑一些 bash 命令扫描环境，把结果注入上下文。

这和我在 Prompt Caching 系列里讲的 **Just-in-Time Context** 思路一致——在正确的时机注入正确的信息，减少 Agent 自行探索的错误面。

### 4.3 修改 3：死循环检测

Agent 有时候会在同一个文件上反复做小修改，10+ 次还在原地打转。他们通过 hook 追踪每个文件的编辑次数，超过阈值就注入"考虑换个方案"的提示。

### 4.4 修改 4：Reasoning Sandwich

还有一个有意思的发现：GPT-5.2-Codex 有 4 档推理强度（low/medium/high/xhigh），全程 xhigh 反而得分低（53.9%），因为超时了。最终他们用 xhigh → high → xhigh 的"三明治"策略——规划和验证用高推理，执行阶段用中等推理。

这四个改动没有一个涉及模型本身。全是环境和流程的变化。


## 5. The Bitter Lesson 再思考（记在🧠里）

[The importance of Agent Harness in 2026](https://www.philschmid.de/agent-harness-2026)文章中有三条建议：

1. **Start Simple**：不要搭复杂的控制流，给好工具，让模型自己规划
2. **Build to Delete**：架构要模块化，新模型会淘汰你昨天写的逻辑
3. **The Harness is the Dataset**：竞争壁垒不是 prompt，是 Harness 捕获的执行轨迹

第三点特别值得展开。Harness 跑的每一次 Agent 任务都在产生数据——成功的路径、失败的模式、工具调用的序列。这些数据可以反馈回训练，让下一代模型更适配 Harness 环境。LangChain 也提到了这一点：模型和 Harness 正在形成共同进化（co-evolution）。

但 The Bitter Lesson 也在起作用。Manus 6 个月重构了 5 次 Harness，Vercel 删掉了 80% 的 Agent 工具反而效果更好。这说明很多当前的 Harness 设计是在弥补模型的不足，模型一进步这些设计就过时了。


Q: 那什么设计是持久的？

我在 Prompt Caching 系列的结尾说过：**围绕 Cache 的架构决策是持久的，因为它们不是在弥补模型不足，而是在[适配当前 Decoder 带来的物理限制](/post/understanding-kv-cache-and-prompt-cache-basics)** 同样的逻辑：文件系统作为持久存储、sandbox 隔离、版本控制——这些是物理约束决定的，不会因为模型变强而消失。


## 6. 总结

Harness Engineering 不是一个全新的发明，而是给一系列已有实践起了一个统一的名字。如果你已经在写 CLAUDE.md、配 MCP Server、用 Sub-agent、做 Context Compaction——你其实已经在做 Harness Engineering 了。

核心就一句话：**Agent 表现不好，先看 Harness 再怪模型。**


## 参考

- OpenAI - [Harness engineering: leveraging Codex in an agent-first world](https://openai.com/index/harness-engineering/) (2026.02)
- LangChain (Vivek Trivedy) - [The Anatomy of an Agent Harness](https://blog.langchain.com/the-anatomy-of-an-agent-harness/) (2026.03)
- LangChain - [Improving Deep Agents with harness engineering](https://blog.langchain.com/improving-deep-agents-with-harness-engineering/) (2026.02)
- HumanLayer - [Skill Issue: Harness Engineering for Coding Agents](https://www.humanlayer.dev/blog/skill-issue-harness-engineering-for-coding-agents) (2026.03)
- 本博客 - [Agent 系统中的 Prompt Caching 设计（上）](/post/prompt-cache-design-for-llm-agents)
- 本博客 - [Agent 系统中的 Prompt Caching 设计（下）](/post/agent-context-management-and-sub-agents)
