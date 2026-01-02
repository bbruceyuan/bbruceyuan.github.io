#!/usr/bin/env python3
"""
自动生成 README.md
扫描 docs/ 目录，提取最近 50 篇文章，生成完整的 README
"""

import re
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict

# ============ 配置 ============
BASE_URL = 'https://yuanchaofa.com'
BLOG_LIMIT = 50

# ============ README 模板 ============
README_TEMPLATE = '''### Hi, I'm chaofa

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

----

**关于我：**
- 🧑‍💻 写代码 @ [github.com/bbruceyuan](https://github.com/bbruceyuan)，使用 Claude Code；个人使用 [ClaudeCode/CodeX API 代理](https://moacode.org/register?ref=bbruceyu)
- 📝 写技术博客 @ [yuanchaofa.com](https://yuanchaofa.com)，专注 LLM、Agent、深度学习
- 🎬 录视频教程 @ [B站](https://space.bilibili.com/12420432) / [YouTube](https://www.youtube.com/@bbruceyuan)，全网「[chaofa用代码打点酱油](https://yuanchaofa.com)」
- 📚 开源项目：[LLMs-Zero-to-Hero](https://github.com/bbruceyuan/LLMs-Zero-to-Hero) - 从零学习大模型
- 💬 交流微信：`bbruceyuan`（请备注来意）

**公众号：**
- 公众号同步更新 Blog 文章
  - <img src="https://yuanchaofa.com/llms-zero-to-hero/chaofa-wechat-official-account.png" width="150" alt="公众号二维码">

----

## 最近文章

{blog_table}
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


def scan_posts(docs_dir: Path) -> List[Dict]:
    """扫描所有博客文章"""
    posts = []

    for subdir in ['blog', 'post', 'hands-on-code', 'llms-zero-to-hero', 'introduction-to-computing-advertising']:
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

                posts.append({
                    'title': title,
                    'date': parsed_date,
                    'permalink': permalink,
                })
            except Exception as e:
                print(f"Error processing {md_file}: {e}")
                continue

    return posts


def generate_table(posts: List[Dict], base_url: str, limit: int) -> str:
    """生成 markdown 表格"""
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


def generate_readme(posts: List[Dict]) -> str:
    """生成完整的 README 内容"""
    blog_table = generate_table(posts, BASE_URL, BLOG_LIMIT)
    return README_TEMPLATE.format(blog_table=blog_table)


def main():
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    docs_dir = project_root / 'docs'
    readme_path = project_root / 'README.md'

    print(f"Scanning posts in {docs_dir}...")
    posts = scan_posts(docs_dir)
    print(f"Found {len(posts)} posts")

    print("Generating README...")
    readme_content = generate_readme(posts)

    print(f"Writing {readme_path}...")
    readme_path.write_text(readme_content, encoding='utf-8')
    print("README.md generated successfully!")


if __name__ == '__main__':
    main()
