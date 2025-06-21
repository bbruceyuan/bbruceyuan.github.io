---
title: "2025-03-走慢些，走远点"
description: 走慢些，走远点。整个 Q1 都处于比较紧张的状态，工作日忙工作，周末学点知识或者做视频，整个人非常地疲劳，但已经明显感觉到不可持续了。而无论是工作、开源项目还是投资，都是需要长期地投入，并且保持耐心，适当的放松是为了走得更远。
date: 2025-03-30 22:18:00
tag:
  - 杂谈
  - chaofa
  - "month-summary"
category:
  - 月度总结
permalink: /blog/2025-03-month-summary.html
---

## 1. 感想

**走慢些，走远点**。

整个 Q1 都处于比较紧张的状态，工作日忙工作，周末学点知识或者做视频，整个人非常地疲劳，但已经明显感觉到不可持续了。而无论是工作、开源项目还是投资，都是需要长期地投入，并且保持耐心，适当的放松是为了走得更远。

## 2. 开源

先说些让人开心的事情，我收获了第一个 **1K Star** 的 GitHub Repo，[LLMs-Zero-to-Hero](https://github.com/bbruceyuan/LLMs-Zero-to-Hero)，这算是这个月最让我有成就感的事情了。

从 15 年 注册 GitHub 依赖，至今快 10 年的时间，断断续续尝试过做一些开源项目，却因寥寥关注导致无疾而终。包括 18 年想做的[ LeetCode Algorithm Repo](https://github.com/bbruceyuan/algorithms-and-oj)，写了一篇知乎文章[《美化你LeetCode仓库的README》](https://zhuanlan.zhihu.com/p/33211817) 至今还有一些阅读，但后面哪怕写题了也没有往上面 push 了；硕士毕业工作后，慢慢开始接触广告算法，也想去贡献当初比较火的 [DeepCTR](https://github.com/shenweichen/DeepCTR)，后面因业务需要，工作中开始做一些召回匹配相关的事情，在一些同事带领下做了一个内部版的 [DeepMatch](https://github.com/shenweichen/DeepMatch)（最终获得了不少兄弟团队的使用，获得腾讯代码铜奖），而我想着 `deepctr` 有 PyTorch 版的，但是 `DeepMatch` 没有，因此开源了一个 [DeepMatch-Torch](https://github.com/bbruceyuan/DeepMatch-Torch)，但这些项目均没有收到什么关注，因此每每想到要做些什么就想着算了吧。

但是自从开始做 BiliBili 视频后，由于视频需要有代码分享，慢慢又把一些代码放到 GitHub，比如  [Hands-On-Large-Language-Models-CN](https://github.com/bbruceyuan/Hands-On-Large-Language-Models-CN), [AI-Interview-Code](https://github.com/bbruceyuan/AI-Interview-Code), [**bit-brain**](https://github.com/bbruceyuan/bit-brain)等，又开始受到一些朋友的关注，信心又开始积累了。此后我将投入更多地时间做一些有意义的项目，比如将 [**bit-brain**](https://github.com/bbruceyuan/bit-brain) 维护成一个超过 [minimind](https://github.com/jingyaogong/minimind) 的项目，因此 minimind 也是一个比较简单的项目，因此将其作为一个终极目标也不是不可行。

## 3. 视频
本周一共更新 4 个视频，但真正算是**这个月录制且有意义**的视频其实只有 2 个，一个是【[从零手写矩阵吸收版的 MLA](https://www.bilibili.com/video/BV1wjQvY6Enm)】，一个是【[OpenManus 源代码的阅读](https://www.bilibili.com/video/BV1SrRhYmEgm)】。当然顺带写了两个文字版文章，一个是 [Kimi K1.5 论文的解读](https://yuanchaofa.com/post/kimi-k1.5-paper-reading-notes.html)，一个是 [MLA 矩阵吸收版](https://yuanchaofa.com/post/hands-on-deepseek-mla-projection-absorption.html)。

这个月我有意识的给自己放松了，因为在工作上我的消耗是一直比较大的，所以我周末其实也很难有能量继续保持学习和输出，因此视频输出放一放也行吧，毕竟这东西除了虚荣心，其他方面带来的价值还是太少了。而且播放其实也不太多，所以真正关注学习的也不多，还是优先自己的学习，顺带输出才行，而不是每周末还需要思考一下这周做一个什么视频。

## 4. 工作

3 月 5 日，[Manus](https://manus.im/) 发布，让 Agent 应用进入到了大众的视野中。尽管理论上来说， Manus 应用的技术可能并不是非常的惊世骇俗，但是展现的效果看上去还是很好的（也许是 Cherry pick 的结果）。

为什么我这么关注这个事情，因为我是团队中 Agent 在业务方向应用的 Owner。Manus 发布自然也会受到产品和老板的关注，那么我就要承接更多地任务和压力，所以在 OpenManus 发布后不久，周末紧急看了一波源代码，就是想借此说明大家使用技术方案可能类似，以及通过开源产品告诉业务方现在 Agent 的水位。现在大家对于 Agent 的预期还是比较高的，但是很多时候，业务上要解决的问题和 Agent 的能力并不一致。比如 Agent 也许能使用好某个工具，但是业务上可能希望使用工具后不能透露工具相关信息，这对于单 Agent 就变得比较复杂；而如果是多 Agent 又比较依赖于工作流（workflow）且延时比较高。

此外还有一个让人心烦的事情，我内推的一个同学因为一些原因竟然没有通过面试，非常地惋惜。我非常看好该同学能力，并且如果入职后可以极大的缓解业务压力，但最终没能入职甚是让我心烦，这让我不得不再次感叹：当我们进入社会之后，做任何事情都隐藏着巨大的随机性和不确定性，无论是面试、工作还是爆款视频，都需要有一定的运气在里面。而我们能做的就是通过自己的努力尽量减少其中的随机性。

## 5. 投资

3 月在投资上做得最大地决策就是慢慢清空了中概，目前富途账户已经是全量现金了（后续转去长桥）。这次不想重蹈 24 年 10 月的老路，慢慢回本，把学费挣回来，距离整体仓位回本还差 7W RMB，年底有希望变正。

![2025-03-走慢些，走远点-20250330225128586](https://cfcdn.bruceyuan.com/blog/2025/2025-03-走慢些，走远点-20250330225128586.webp)


> 下个 Q 要重新开始锻炼了，保持良好的精神状态面对工作和未来。