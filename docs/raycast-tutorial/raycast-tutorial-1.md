---
title: Raycast使用指南（一）--基本用法
id: 36
date: 2023-02-11 15:06:00
description: Raycast使用指南第一篇，介绍raycast基本用法，包括文件应用查找，剪贴板历史，窗口管理等功能
category: 
  - raycast-turorial
permalink: /post/raycast-tutorial-1.html/
head:
  - - meta
    - name: keywords
      content: raycast tutorial, raycast 用法, raycast 简介
---

尽管目前网络上有很多的推荐 Raycast[^1] 的文章，也有很丰富的教程和链接，但是我觉得都没有很**系统地介绍 Raycast 功能**。可能是主要是觉得缺少一个中文的文档吧。如果读者英文已经很好了，其实也没有必要读接下来的文章，直接看[官方文档](https://manual.raycast.com/) [^2]会是一个更完整更可靠的选择。

我已经差不多使用 Raycast 一年了，一开始是被 **Raycast 颜值** 所吸引，然后是深深的感受到它的**便利性**，最后因为最近使用 Raycast 的脚本扩展所折服。因此想简单**写一个 Raycast 的介绍文章，让更多的人可以感受到它的便利性**。


## 1. 什么是 Raycast？
Raycast 是一个在 **Mac 上的启动器**，类似于 Mac 自带的 「焦点（Spotlight）」。所谓的启动器（Launcher），可以理解为管理各种电脑应用（Application）的的工具，让你更加**高效的组织应用并且与应用交互**。

Raycast 作为一个很好用的启动器，具有很多基础的功能，比如查找应用、查找文件、使用计算器、历史剪切板等；除了基础功能以外，Raycast 还支持插件扩展，Raycast 的插件和一些脚本都是[开源](https://github.com/raycast/extensions)的，因此你可以根据自己的需求在[插件市场](https://www.raycast.com/store)和[脚本仓库](https://github.com/raycast/script-commands)中找到自己想要的插件和脚本；如果在插件/脚本市场中看了一圈，没有开源项目可以满足你需要的功能，那么你可以自己写一个插件或者通过一些脚本（Script）满足自己的要求。

RayCast 一般运行于后台，只有在你需要的时候找它就OK了，其他时候可以用完就走。而我们一般都会设置一个快捷键，通常都是设置成  **`command + 空格（space）`** 唤醒，这样可以同时禁止掉 Mac 自带的 Spotlight。

## 2.  Raycast 基本结构
首先通过 **`command + space`** 打开 Raycast。 基本结构如下，整体比较简洁。

![Pasted image 20230203163513.png](/blog_imgs/raycast/raycast-1.png)

> 上图或者下文提到的 actions 可以理解为 **「更多操作」**。

上图展示了 Raycast 的全貌。可以说非常的美观。为了让 Raycast 更实用，我们也可以在 Raycast 菜单栏**为各种应用设置快捷键和别名**，因此我们可以通过别名的方式让 **Raycast 支持拼音搜索**。具体结构如下：

![Pasted image 20230203174504.png](/blog_imgs/raycast/raycast-2.png)


## 3.  基本功能
了解了 Raycast 的基本功能，就掌握了它 90% 的功能了，基本满足日常使用需求了。因此下面我按照日常使用频次对相关功能进行了排序。

此外一定要记住：**不要怕记不住这些命令，raycast 是支持模糊搜索的，你知道个大概就能匹配出来。如果实在记不到复杂的，可以把常使用的加上别名，通过拼音就没有问题了。**

### 2.1 查找内容（应用、文件等）
在输入框中的输入你想要的搜索的东西就可以查找对应的东西。

-  **搜索应用**

比如查找**微信**，可以直接点击回车打开微信。当然有的时候你可能是想查找文件名是微信的文件，因此他也会推荐你使用「file search（文件搜索）」打开，这样你就可以查看文件名中有微信的文件了。

![Pasted image 20230203164010.png](/blog_imgs/raycast/raycast-3.png)

- **搜索文件**

此外，如果想直接搜索文件，最好是**先输入file或者 file search**，然后通过 `tab` 建或者 回车键进入 file search 插件（当然这里也可以设置快捷键）；

step 1:  进入插件

![Pasted image 20230203164944.png](/blog_imgs/raycast/raycast-4.png)

step 2：搜索内容。可以得到相应的结果内容，右边还会有预览信息。如果只是想直接打开，也可以使用 **`command + k`** 打开更多选项，比如 **「复制、剪切、在文件夹中打开」** 等。

![Pasted image 20230203165431.png](/blog_imgs/raycast/raycast-5.png)

### 2.2 剪切板历史
进入剪切板可以通过输入 **「clipboard history、history」** 或者使用快捷键进入，类似上面的文件查找；使用方式也和文件查找是一样的。

![Pasted image 20230203170714.png](/blog_imgs/raycast/raycast-6.png)

剪切板可以在左下角按钮中设置保留日期、匹配方式和预览方式等内容。

> Q：为什么剪切板历史很有用？  
> 
> A：可以极大的提升工作效率。我们很多时候，都不止需要复制一个东西，如果在不同的APP之间剪切+复制，就会很烦。 但是有了剪切板历史之后，就可以一次性把要复制的内容都复制下来，生成几条复制历史，然后到对应的APP中取粘贴，这个体验是真的好。


### 2.3 计算器小功能
计算器用的也是比较多。直接输入相关的数字做**数学计算**，也可以**计算汇率**。

![Pasted image 20230203170024.png](/blog_imgs/raycast/raycast-7.png)

除了这些计算小功能之外，还可以在计算器中 **查看今天是几号**，计算 **「今天和 5月8号差几天」** 等；更多例子[访问原文](https://manual.raycast.com/calculator)。

### 2.4 系统功能（锁屏重启、亮度调节、音量控制等）
以下列举几个常见我经常使用的命令；注意这些命令是可以模糊匹配的，因此不用因为记不住完整命令而苦恼：
- 清空回车站； **Empty Trash**
- 锁屏；**Lock Screen**
- 重启；**Restart**
- 睡眠；**Sleep**
- 静音；**Tongle Mute**

更多命令可以看 [原文](https://manual.raycast.com/system)，也可以查看这个[视频](https://super-static-assets.s3.amazonaws.com/1c0bdccb-1297-47fd-9b2d-851c4e24e453/videos/fc51e02e-c451-45c1-9625-a476f7a25d4d.mp4)。

### 2.5 窗口管理
可能很多人没有用过窗口管理软件，这可能是因为在 windows/mac 中，自带了很多窗口管理功能。比如**双击应用顶部栏，就可以全屏**，这是一种最简单的窗口管理。

如果有更复杂一点呢？
1. **两个应用重叠了，想两个分别占半屏**，这种时候普通的点击就做不到了，自己拖拽也很多完成。但是有了「窗口管理」之后，就可以轻松在raycast中输入命令。`left half` 和 `right half`轻松完成布局。
2. 想**快速全屏**，有不想动鼠标；输入 `Maximize` 就可以了。
3. 想**快速得到一个合理的窗口大小**，用鼠标拖拽也不一定精确。输入 `Reasonable size`，这样可以让当前窗口占屏幕的 60%。非常的好用。

此外，还可以通过命令完成 **「上移、下移、移动到最顶部、窗口居中」** 等功能，更多命令可以看[原文](https://manual.raycast.com/window-management)。

### 2.6 其他
raycast 还有很多其他的内置功能，我用的就比较少了，因此这里也不做过多的介绍。

自带功能：
- 创建 Snippets，**snipets**
	- Snippets 指一些通用、常见的小片段，然后通过设置别名来快速键入。有点类似输入法中的常用语。
- 常看日历中的事件，**calendar**
- 在日历中创建时间，**Reminders**

还有一些[团队相关的功能](https://manual.raycast.com/getting-started-with-teams)没有用，感兴趣的同学也可以自行查看。

## 4. 常见问答（FAQ）
- 可以使用「焦点（spotlight）」的快捷点来启动 raycast吗？
	- 可以。在设置中把快捷点替换掉就OK了。

## 5. Reference
[^1]: https://www.raycast.com/
[^2]: https://manual.raycast.com/
