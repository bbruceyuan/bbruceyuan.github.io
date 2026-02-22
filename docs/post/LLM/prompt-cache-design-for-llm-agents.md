---
title: Agent 系统中的 Prompt Caching 设计（上）：Cache 破坏、Prompt 布局与工具管理
date: 2026-02-22 10:16:00
tag:
  - agent
  - LLM
  - KV Cache
  - Context Engineering
category:
  - hands-on-code
description: 深入分析 AI Agent 为什么比 Chatbot 更需要 Prompt Caching，什么操作会破坏 Cache，以及 Claude Code、Manus、OpenAI Codex 在 Prompt 布局和工具管理上的 Cache-aware 设计实践。
publish: true
permalink: /post/prompt-cache-design-for-llm-agents.html
---

## 0. 阅读收获 (takeaway)

读完本文，你将了解：

- 从 Prompt Engineering 到 Context Engineering 的范式转变
- 为什么 Agent 比普通 Chatbot **更**需要 Prompt Caching
- 什么操作会破坏 Cache（比你想象的多）
- Prompt 布局与动态信息管理的最佳实践
- 工具管理的三种 Cache-aware 方案对比

> **前置知识**：本文假设你已经理解 KV Cache、Prefill/Decode 两阶段、以及 Prompt Cache 的前缀匹配机制。如果不熟悉这些概念，建议先阅读 [理解 KV Cache 与 Prompt Caching：LLM 推理加速的核心机制](/post/understanding-kv-cache-and-prompt-cache-basics.html)。

## 1. 先说结论：Cache Rules Everything

在深入细节之前，我想先分享我对这个话题的理解：

> **Prompt Cache 不只是一个省钱技巧，它是 Agent 系统架构设计的核心约束。**

就像数据库的 schema 设计会影响整个应用架构一样，Prompt Cache 的前缀匹配约束深刻地影响了 Agent 的每一个设计决策：

- prompt 怎么组织？→ 稳定内容放前面，变化内容放后面
- 工具怎么管理？→ 工具列表固定，通过其他机制限制可用范围
- 状态怎么切换？→ 不切换工具，用工具模拟状态转换（Claude Code Plan Mode 就是最好的例子）
- context 怎么压缩？→ 压缩操作本身必须 cache-safe
- 模型怎么选？→ 不在同一会话中切换，用子代理隔离

### 1.1 从 Prompt Engineering 到 Context Engineering

我们已经越来越多地听到 "Context Engineering" 这个术语。区别在哪？

- **Prompt Engineering** 关注 "怎么写指令让模型表现更好"——内容层面的优化。
- **Context Engineering** 关注 "怎么组织整个上下文——指令、工具、历史、外部信息——让 Agent 系统整体高效运转"——系统架构层面的设计。

[Manus 在 25 年底最新总结](https://rlancemartin.github.io/2025/10/15/manus/)中提出了三个维度：**Reduce**（缩减）、**Isolate**（隔离）、**Offload**（卸载）。

后面的内容你会看到，各家的设计都在围绕这三个维度展开。

### 1.2 三家方案的共同规律

不同公司、不同架构，但核心规律惊人地一致：

1. **前缀不变**：system prompt、tools、早期历史永远不修改
2. **追加不修改**：Append-only，永远不编辑历史消息
3. **工具定义稳定**：tools 数组不变，通过其他机制控制可用范围
4. **动态信息后置**：时间戳、环境状态等放在后面的 user message 中
5. **压缩必须 cache-safe**：压缩操作复用父对话的 cache prefix

带着这些规律，我们来看具体的实践。

## 2. 为什么 Agent 比 Chatbot 更需要 Prompt Cache？

### 2.1 Agent 的 I/O 比例严重失衡

Agent 每一步都需要发送完整的对话历史给模型，模型只输出一小段。Manus 披露过一个数据：**input:output ≈ 100:1**。

如果没有 Prompt Cache → 每一步重新 Prefill 所有历史 token → **成本二次方增长**。

### 2.2 经济账

| 场景 | 不缓存 | 缓存后 | 节约 |
|------|--------|--------|------|
| Claude（正常 vs cached）| $3/MTok | $0.30/MTok | **90%** |
| OpenAI GPT-5（正常 vs cached）| $10/MTok | $2.50/MTok | **75%** |
| Claude Code 单任务（约 2M tokens）| ~$6.00 | ~$1.15 | **81%** |

Thariq（Claude Code 团队）：

> "Coding agents would be cost prohibitive without prompt caching."

OpenAI Codex：cache 命中后，**采样开销从二次降为线性**。

### 2.3 延迟：TTFT 的大幅改善

- **Manus**：KV cache hit rate 是 "**the single most important metric**"
- **Claude Code 团队**：cache 命中率下降 → 当作**线上事故**（SEV）处理
- **OpenAI**：150K+ tokens 时，cached 请求 TTFT 快 **67%**

## 3. 什么操作会破坏 Cache？

前缀匹配是一切的基础：**任何位置的任何改动 → 该位置之后的 cache 全部失效。**

### 3.1 改动 System Prompt

**在开头放时间戳**——Manus 踩过的坑。时间戳每秒都变，第一个 token 就不同，整个 cache 废掉。

### 3.2 改动 Tool Definitions

Claude Code 团队经验中**最常见的 cache 破坏方式**。Tool definitions 在 prompt 前部，增删任何一个工具 → 后续所有 cache 失效。

具体场景：
- **增删工具**：动态加载不同工具集 → 每次变化 cache 全废
- **工具顺序不确定**：Codex 的 MCP 工具注册顺序不确定
- **更新工具参数**：如 `allowed_subagents` 列表变化

### 3.3 切换模型

Cache 是 **model-specific** 的。

一个反直觉推论：**100K token 对话中，切换到更便宜的模型可能更贵** —— Opus 的 100K cached token 只需 $1.50，换 Haiku 后全部重算。

### 3.4 修改历史消息

- 编辑或删除之前的 action/observation → 破坏 cache
- **非确定性序列化**：JSON key 排序不一致（Manus 踩坑）→ 相同语义不同 token 序列

> 核心原则：**序列化必须是确定性的。**

## 4. Prompt 布局与动态信息管理

核心思路：**把稳定的内容放前面，把变化的内容放后面。**

### 4.1 Claude Code 的四层缓存架构

![Claude Code 四层布局与 Codex Prompt 构建过程对比](/blog_imgs/agent-prompt-layout.png)

| 层级 | 内容 | 稳定性 |
|------|------|--------|
| Layer 1 | Static System Prompt & Tools | 全局不变 |
| Layer 2 | CLAUDE.md 项目配置 | 项目级不变 |
| Layer 3 | Session Context（git status 等） | 会话级 |
| Layer 4 | Conversation Messages | 每轮追加 |

每轮只有 Layer 4 增长，前 3 层稳定命中 cache。

### 4.2 OpenAI Codex 的 Prompt 构建

三层结构：**instructions → tools → input**（input 可能会包含 dev role message，上图）。关键设计：**旧提示是新提示的精确前缀**。

配置变更（沙盒权限、工作目录）→ **追加**新消息而非**修改**旧消息。

### 4.3 Manus 的三条规则

1. **稳定前缀** 
2. **Append-only** 
3. **确定性序列化**

### 4.4 动态信息怎么更新？

| 方案 | 实现方式 |
|------|---------|
| Claude Code | `<system-reminder>` 标签放在 user message 中 |
| Codex | 追加新的 developer/user 消息 |

> **永远追加，永远不修改。**

### 4.5 Cache Breakpoint 与 Auto-caching

- **Claude API**：从手动 `cache_control` breakpoint → **auto-caching** 一个参数搞定
- **OpenAI API**：`prompt_cache_key` 路由优化，自动缓存 ≥1024 token 前缀

反直觉：900 token prompt 永远不 cache hit，扩展到 1024+ token 反而更省钱。

## 5. 工具管理：三种方案，殊途同归

### 5.1 问题本质

Agent 可能有 30 个工具，但不同阶段只需要一部分。如果按需加载 → 每次状态切换 cache 全废。

![三家工具管理策略对比](/blog_imgs/tool-management-comparison.png)

### 5.2 Claude Code：状态转换 + defer_loading

- **Plan Mode**：`EnterPlanMode`/`ExitPlanMode` 本身作为工具 → 工具列表永远不变
- **Tool Search**：`defer_loading` stub → `ToolSearch` 按需获取完整 schema
- 模型可**自主决定**何时进入 Plan Mode

### 5.3 Manus：Logits Masking

- 所有工具始终在 prompt 中
- 工具命名约定：`browser_xxx`、`shell_xxx`
- Token logits masking 控制可用工具
- 三种模式：Auto / Required / Specified

### 5.4 OpenAI：allowed_tools 参数

- `tools` 数组完整不变
- `allowed_tools` 限制当前可用子集
- 注意：MCP 服务器可动态变更工具列表 → 需谨慎处理

### 5.5 对比总结

> **本质：工具定义不变（保 cache），通过其他机制限制可选范围。**

| 方案 | 实现方式 | 优点 | 限制 |
|------|---------|------|------|
| Claude Code | tool 本身 + defer_loading | 灵活，模型自主决策 | 需 API 支持 |
| Manus | logits masking | 精细控制 | 需 self-hosting |
| OpenAI | allowed_tools 参数 | 最简单 | 仅粗粒度 |

## 下一篇

本文聚焦于 Cache-aware 的 Prompt 设计和工具管理。但 Agent 还面临另一组挑战：context 越来越长怎么办？怎么压缩才不破坏 cache？子代理怎么设计？

下一篇 [Agent 系统中的 Prompt Cache 设计（下）：上下文管理与子代理架构](/post/agent-context-management-and-sub-agents.html) 将深入这些话题。

## 参考

- Thariq @trq212 - [Lessons from Building Claude Code: Prompt Caching Is Everything](https://x.com/trq212)
- Lance Martin @RLanceMartin - [Prompt auto-caching with Claude](https://blog.langchain.dev/prompt-auto-caching-with-claude/)
- Manus Blog - [Context Engineering for AI Agents](https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus)
- Michael Bolin - [深入解析 Codex 智能体循环](https://openai.com/index/building-the-codex-agent-loop/)
- [Prompt Caching 201](https://cookbook.openai.com/examples/prompt_caching_201)
- [Effective Context Engineering for AI Agents](https://www.anthropic.com/engineering/context-engineering)
- [理解 KV Cache 与 Prompt Cache](/post/understanding-kv-cache-and-prompt-cache-basics.html) - 我的博客

## 其他

最后欢迎关注我，基本全网同名 [chaofa用代码打点酱油](https://yuanchaofa.com/)

- 公众号（主要是为了订阅通知，不然看 Blog 就够了）： ![chaofa用代码打点酱油](https://yuanchaofa.com/llms-zero-to-hero/chaofa-wechat-official-account.png)
- [B站-chaofa用代码打点酱油](https://space.bilibili.com/12420432)
- [YouTube-chaofa用代码打点酱油](https://www.youtube.com/@bbruceyuan)
- [chaofa 的 notion 简介](https://chaofa.notion.site/11a569b3ecce49b2826d679f5e2fdb54)
