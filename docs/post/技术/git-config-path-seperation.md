---
title: 项目隔离，不同的项目使用不同的 Git 配置
date: 2024-08-30 23:20:20
tag:
  - git
description: 利用 Git 的 includeIf 配置，为不同的项目设置不同的配置，通过子文件夹 .gitconfig 覆盖 Git 的 Global 相关配置，比如 name，email 等。
publish: true
permalink: /post/git-config-path-seperation.html
---

## 适应场景

- 情况 1：你可能有多个邮箱，你希望项目 A 用邮箱 A，项目 B 用邮箱 B。很直观的做法就是在每个不同的项目中设置不同的邮箱。每次 `git clone xxx` 项目之后，`git config user.name xxx` 和 `git config user.email xxx`，但是每个项目都要自己设置就非常的麻烦。
- 情况 2：你同时有公司的项目和自己的项目，希望公司的项目用公司邮箱，自己的项目用自己的邮箱，那么这种情况下一般会先设置 `global`的 `name` 和 `email`为公司邮箱，**固定某个文件夹**（personal_projects）设置为自己的邮箱。这种方法的优点是不需要在每个仓库中单独设置 Git 配置，而且可以轻松地管理多个项目或文件夹的不同 Git 身份。

## 配置步骤

我们可以使用 Git 的条件包含（conditional includes）功能来为特定文件夹设置不同的 `user.name` 和 `user.email`。

1. 首先，全局 Git 配置中设置默认的 `user.name` 和 `user.email`（如果还没有设置的话）：

   ```bash
   git config --global user.name "bbruceyuan_glocal"
   git config --global user.email "bbruceyuan_glocal.email@example.com"
   ```

1. 在 `personal_projects` 文件夹中创建一个 `.gitconfig` 文件：

   ```bash
   cd path/to/personal_projects
   touch .gitconfig
   ```

1. 编辑 `personal_projects/.gitconfig` 文件，添加以下内容：

   ```ini
   [user]
       name = bbruceyuan
       email = bruceyuan@mail.com
   ```

1. 在全局 `.gitconfig` 文件中（通常位于 `~/.gitconfig`）添加条件包含配置：

   ```ini
   [includeIf "gitdir:/path/to/personal_projects/"]
       path = /path/to/personal_projects/.gitconfig
   ```

   注意：

   - 路径必须以斜杠 `/` 结尾。
   - 路径可以是绝对路径 `/path/to/personal_projects/`
   - 路径可以是相对路径，例如 `~/personal_projects/person/`。

这样设置后，`Git` 将在 `personal_projects` **文件夹及其子文件夹**中使用特定的 `user.name (bbruceyuan)` 和 `user.email (bruceyuan@mail.com)`，而在其他地方使用全局设置。
