#!/usr/bin/env python3
"""
自动生成两个 README.md：
1. bbruceyuan.github.io 的 README（最近 50 篇全部文章）
2. bbruceyuan 的 README（技术 5 篇 + 生活 5 篇，四列表格）
"""

import re
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict

# ============ 配置 ============
BASE_URL = 'https://yuanchaofa.com'
BLOG_LIMIT = 50
PROFILE_LIMIT = 5

# ============ bbruceyuan.github.io README 模板 ============
BLOG_README_TEMPLATE = '''### Hi, I'm chaofa

> chaofa用代码打点酱油 | LLM & Agent 爱好者 | 技术博主 | 瞎折腾爱好者

<p align="center">
  <a href="https://yuanchaofa.com"><img src="https://img.shields.io/badge/Blog-yuanchaofa.com-pink?style=flat-square" /></a>
  &nbsp;&nbsp;
  <a href="https://space.bilibili.com/12420432"><img src="https://img.shields.io/badge/Bilibili-chaofa-ff69b4?style=flat-square&logo=bilibili&logoColor=white" /></a>
  &nbsp;&nbsp;
  <a href="https://www.youtube.com/@bbruceyuan"><img src="https://img.shields.io/badge/YouTube-chaofa-red?style=flat-square&logo=youtube&logoColor=white" /></a>
</p>

<p align="center">
  <a href="https://www.zhihu.com/people/bbruceyuan"><img src="https://img.shields.io/badge/知乎-bbruceyuan-blue?style=flat-square&logo=zhihu&logoColor=white" /></a>
  &nbsp;&nbsp;
  <a href="https://github.com/bbruceyuan/LLMs-Zero-to-Hero"><img src="https://img.shields.io/badge/LLMs--Zero--to--Hero-教程-green?style=flat-square&logo=github" /></a>
  &nbsp;&nbsp;
  <img src="https://komarev.com/ghpvc/?username=bbruceyuan&label=Views&color=0e75b6&style=flat-square" alt="访问量" />
</p>

**关于我：**
- 🧑‍💻 Vibe Coding @ [github.com/bbruceyuan](https://github.com/bbruceyuan)
  - 使用 [Claude Code、CodeX](https://moacode.org/register?ref=bbruceyu)、Cursor 等
  - 个人推荐 [ClaudeCode/CodeX API 代理](https://moacode.org/register?ref=bbruceyu) 获取更好的 vibe 体验
- 📝 写技术博客 @ [yuanchaofa.com](https://yuanchaofa.com)，专注 LLM、Agent、深度学习
- 🎬 录视频教程 @ [B站](https://space.bilibili.com/12420432) / [YouTube](https://www.youtube.com/@bbruceyuan) / [视频号](https://yuanchaofa.com/llms-zero-to-hero/chaofa-wechat-official-account.png)，全网同名「[chaofa用代码打点酱油](https://yuanchaofa.com)」
- 📚 开源项目：[LLMs-Zero-to-Hero](https://github.com/bbruceyuan/LLMs-Zero-to-Hero) - 从零学习大模型
- 💬 交流微信：`bbruceyuan`（请备注来意）
- **公众号：** 公众号同步更新 Blog 文章
  - <img src="https://yuanchaofa.com/llms-zero-to-hero/chaofa-wechat-official-account.png" width="200" alt="公众号二维码">

----

## 最近文章

{blog_table}
'''

# ============ bbruceyuan（GitHub Profile）README 模板 ============
PROFILE_README_TEMPLATE = '''### Hi there 👋

<div align="center">

  <!-- for beauty 留个空行好看点 -->
  <div>&nbsp;</div>

  <!-- profile logo 个人资料徽标 -->
  <div>
    <a href="https://yuanchaofa.com"><img src="https://img.shields.io/badge/yuanchaofa.com-个人博客-pink" /></a>&emsp;
    <a href="https://space.bilibili.com/12420432"><img src="https://img.shields.io/badge/chaofa用代码打点酱油-Bilibili-ff69b4" /></a>&emsp;
    <a href="https://www.youtube.com/@bbruceyuan"><img src="https://img.shields.io/badge/chaofa用代码打点酱油-YouTube-red" /></a>&emsp;
    <br><br>
    <a href="https://www.zhihu.com/people/bbruceyuan"><img src="https://img.shields.io/badge/bbruceyuan-Zhihu-blue" /></a>&emsp;
    <a href="https://yuanchaofa.com/llms-zero-to-hero/chaofa-wechat-official-account.png"><img src="https://img.shields.io/badge/chaofa用代码打点酱油-公众号-green" /></a>&emsp;
    <a href="https://yuanchaofa.com/llms-zero-to-hero/wechat-account-bbruceyuan.png"><img src="https://img.shields.io/badge/bbruceyuan-交流微信（请备注）-green" /></a>&emsp;
    <!-- visitor statistics logo 访问量统计徽标 -->
    <img src="https://komarev.com/ghpvc/?username=bbruceyuan&label=Views&color=0e75b6&style=flat" alt="访问量统计" />
  </div>
</div>
<br />

**关于我：**
- 🧑‍💻 Vibe Coding @ [github.com/bbruceyuan](https://github.com/bbruceyuan)
  - 使用 [Claude Code、CodeX](https://moacode.org/register?ref=bbruceyu)、Cursor 等
  - 个人推荐 [ClaudeCode/CodeX API 代理](https://moacode.org/register?ref=bbruceyu) 获取更好的 vibe 体验
- 📝 写技术博客 @ [yuanchaofa.com](https://yuanchaofa.com)，专注 LLM、Agent、深度学习
- 🎬 录视频教程 @ [B站](https://space.bilibili.com/12420432) / [YouTube](https://www.youtube.com/@bbruceyuan) / [视频号](https://yuanchaofa.com/llms-zero-to-hero/chaofa-wechat-official-account.png)，全网同名「[chaofa用代码打点酱油](https://yuanchaofa.com)」
- 📚 开源项目：[LLMs-Zero-to-Hero](https://github.com/bbruceyuan/LLMs-Zero-to-Hero) - 从零学习大模型
- 💬 交流微信：`bbruceyuan`（请备注来意）

----

## 最近更新

{four_column_table}

> [!NOTE]
> 更多文章请访问
> - 个人 blog [https://yuanchaofa.com](https://yuanchaofa.com)
> - 个人公众号 [chaofa用代码打点酱油](https://yuanchaofa.com/llms-zero-to-hero/chaofa-wechat-official-account.png)
'''


def parse_frontmatter(content: str) -> Optional[Dict]:
    """解析 markdown 文件的 YAML frontmatter"""
    match = re.match(r'^---\s*\n(.*?)\n---\s*\n', content, re.DOTALL)
    if not match:
        return None

    frontmatter_text = match.group(1)
    result = {}

    # 提取 title（可能带引号）
    title_match = re.search(r'^title:\s*["\']?(.*?)["\']?\s*$', frontmatter_text, re.MULTILINE)
    if title_match:
        result['title'] = title_match.group(1).strip().strip('"\'')

    # 提取 date
    date_match = re.search(r'^date:\s*(.+?)$', frontmatter_text, re.MULTILINE)
    if date_match:
        result['date'] = date_match.group(1).strip()

    # 提取 permalink
    permalink_match = re.search(r'^permalink:\s*(.+?)$', frontmatter_text, re.MULTILINE)
    if permalink_match:
        result['permalink'] = permalink_match.group(1).strip()

    # 提取 publish 状态（默认为 true）
    publish_match = re.search(r'^publish:\s*(true|false)', frontmatter_text, re.MULTILINE | re.IGNORECASE)
    if publish_match:
        result['publish'] = publish_match.group(1).lower() == 'true'
    else:
        result['publish'] = True

    return result


def parse_date(date_str: str) -> Optional[datetime]:
    """解析多种日期格式"""
    formats = [
        '%Y-%m-%d %H:%M:%S',
        '%Y-%m-%dT%H:%M:%S',
        '%Y-%m-%d',
    ]

    date_str = date_str.strip().strip('"\'')

    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue

    # 处理月份/日期无补零的情况
    try:
        parts = date_str.replace('T', ' ').split(' ')
        if len(parts) >= 1:
            date_parts = parts[0].split('-')
            if len(date_parts) == 3:
                year, month, day = date_parts
                normalized_date = f"{year}-{int(month):02d}-{int(day):02d}"
                if len(parts) > 1:
                    normalized_date += f" {parts[1]}"
                for fmt in formats:
                    try:
                        return datetime.strptime(normalized_date, fmt)
                    except ValueError:
                        continue
    except (ValueError, IndexError):
        pass

    return None


def scan_posts_by_category(docs_dir: Path) -> Dict[str, List[Dict]]:
    """扫描博客文章，按类别分类"""
    # 技术文章目录
    tech_dirs = ['post', 'hands-on-code', 'llms-zero-to-hero', 'introduction-to-computing-advertising']
    # 个人生活目录
    life_dirs = ['blog']

    result = {
        'tech': [],
        'life': [],
        'all': []
    }

    all_dirs = {'tech': tech_dirs, 'life': life_dirs}

    for category, dirs in all_dirs.items():
        for subdir in dirs:
            target_dir = docs_dir / subdir
            if not target_dir.exists():
                continue

            for md_file in target_dir.rglob('*.md'):
                if md_file.name.lower() == 'readme.md':
                    continue

                try:
                    content = md_file.read_text(encoding='utf-8')
                    metadata = parse_frontmatter(content)

                    if not metadata:
                        continue

                    if not metadata.get('publish', True):
                        continue

                    title = metadata.get('title')
                    date_str = metadata.get('date')
                    permalink = metadata.get('permalink')

                    if not title or not date_str:
                        continue

                    parsed_date = parse_date(date_str)
                    if not parsed_date:
                        print(f"Warning: Cannot parse date '{date_str}' in {md_file}")
                        continue

                    if not permalink:
                        relative_path = md_file.relative_to(docs_dir)
                        permalink = '/' + str(relative_path).replace('.md', '.html')

                    post = {
                        'title': title,
                        'date': parsed_date,
                        'permalink': permalink,
                        'category': category,
                    }
                    result[category].append(post)
                    result['all'].append(post)

                except Exception as e:
                    print(f"Error processing {md_file}: {e}")
                    continue

    return result


def generate_table(posts: List[Dict], base_url: str, limit: int) -> str:
    """生成两列 markdown 表格"""
    sorted_posts = sorted(posts, key=lambda x: x['date'], reverse=True)[:limit]

    lines = [
        '| 日期 | 文章 |',
        '|------|------|',
    ]

    for post in sorted_posts:
        date_str = post['date'].strftime('%Y-%m-%d')
        title = post['title']
        url = base_url + post['permalink']
        lines.append(f'| {date_str} | [{title}]({url}) |')

    return '\n'.join(lines)


def generate_two_column_table(tech_posts: List[Dict], life_posts: List[Dict], base_url: str, limit: int) -> str:
    """生成两列 markdown 表格：技术文章 | 非技术文章，每行格式为 时间 + 标题"""
    sorted_tech = sorted(tech_posts, key=lambda x: x['date'], reverse=True)[:limit]
    sorted_life = sorted(life_posts, key=lambda x: x['date'], reverse=True)[:limit]

    lines = [
        '| 技术文章 | 非技术文章 |',
        '|----------|------------|',
    ]

    max_len = max(len(sorted_tech), len(sorted_life))

    for i in range(max_len):
        # 技术文章列
        if i < len(sorted_tech):
            tech = sorted_tech[i]
            tech_date = tech['date'].strftime('%Y-%m-%d')
            tech_title = tech['title']
            tech_url = base_url + tech['permalink']
            tech_cell = f"{tech_date} [{tech_title}]({tech_url})"
        else:
            tech_cell = ""

        # 非技术文章列
        if i < len(sorted_life):
            life = sorted_life[i]
            life_date = life['date'].strftime('%Y-%m-%d')
            life_title = life['title']
            life_url = base_url + life['permalink']
            life_cell = f"{life_date} [{life_title}]({life_url})"
        else:
            life_cell = ""

        lines.append(f'| {tech_cell} | {life_cell} |')

    return '\n'.join(lines)


def generate_blog_readme(posts: List[Dict]) -> str:
    """生成 bbruceyuan.github.io 的 README"""
    blog_table = generate_table(posts, BASE_URL, BLOG_LIMIT)
    return BLOG_README_TEMPLATE.format(blog_table=blog_table)


def generate_profile_readme(tech_posts: List[Dict], life_posts: List[Dict]) -> str:
    """生成 bbruceyuan（GitHub Profile）的 README"""
    two_column_table = generate_two_column_table(tech_posts, life_posts, BASE_URL, PROFILE_LIMIT)
    return PROFILE_README_TEMPLATE.format(four_column_table=two_column_table)


def main():
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    docs_dir = project_root / 'docs'

    # 输出路径
    blog_readme_path = project_root / 'README.md'
    profile_readme_path = project_root / 'profile_README.md'  # 临时存放，后续推送到 bbruceyuan 仓库

    print(f"Scanning posts in {docs_dir}...")
    posts_by_category = scan_posts_by_category(docs_dir)

    print(f"Found {len(posts_by_category['all'])} total posts")
    print(f"  - Tech posts: {len(posts_by_category['tech'])}")
    print(f"  - Life posts: {len(posts_by_category['life'])}")

    # 生成 bbruceyuan.github.io README
    print("\nGenerating bbruceyuan.github.io README...")
    blog_readme = generate_blog_readme(posts_by_category['all'])
    blog_readme_path.write_text(blog_readme, encoding='utf-8')
    print(f"Written to {blog_readme_path}")

    # 生成 bbruceyuan（Profile）README
    print("\nGenerating bbruceyuan (Profile) README...")
    profile_readme = generate_profile_readme(
        posts_by_category['tech'],
        posts_by_category['life']
    )
    profile_readme_path.write_text(profile_readme, encoding='utf-8')
    print(f"Written to {profile_readme_path}")

    print("\nDone!")


if __name__ == '__main__':
    main()
