import { hopeTheme } from "vuepress-theme-hope";
import nav from './nav';
import sidebar from './sidebar';

export default hopeTheme({
    darkmode: "toggle",
    logo: "/img/icon.jpg",
    print: false,
    pure: true,
    repo: "bbruceyuan",

    // 假如你的文档仓库和项目本身不在一个仓库:
    docsRepo: "bbruceyuan/bbruceyuan.github.io",
    // 假如文档不是放在仓库的根目录下:
    docsDir: "docs",
    // 假如文档放在一个特定的分支下，默认为 'main':
    docsBranch: "main",
    // 默认是 false, 设置为 true 来启用
    contributors: false,

    hotReload: true,
    navbar: nav,
    navbarLayout: {
        start: ["Brand"],
        center: [],
        end: ["Links", "Repo", "Search", "Outlook"]
    },
    sidebar: sidebar,
    hostname: "https://bbruceyuan.com",
    favicon: "/img/icon.jpg",

    author: {
        name: "bbruceyuan",
        url: 'https://bbruceyuan.com',
    },
    blog: {
        description: "想要开始做一点有意义的事情~",
        medias: {
            Github: "https://github.com/bbruceyuan",
            BiliBili: "https://space.bilibili.com/12420432",
            Twitter: "https://twitter.com/BBruceyuan"
        },
    },
    plugins: {
        comment: {
            provider: "Giscus",
            repo: "bbruceyuan/bbruceyuan.github.io",
            repoId: "MDEwOlJlcG9zaXRvcnkyNDU3Njc0NzU=",
            category: "Announcements",
            categoryId: "DIC_kwDODqYdM84Cbg9U",
        },
        blog: {
            excerptLength: 0,
            excerpt: false,
        },
        mdEnhance: {
            // 使用 KaTeX 启用 TeX 支持
            //   katex: true,
            mathjax: true,
            footnote: true,
            tasklist: true,
            imgLazyload: true,
        },
    },
    headerDepth: 3
})
