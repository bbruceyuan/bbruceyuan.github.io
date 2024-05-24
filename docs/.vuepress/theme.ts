import { hopeTheme } from "vuepress-theme-hope";
import nav from './nav';
import sidebar from './sidebar';

export default hopeTheme({
    darkmode: "toggle",
    logo: "/img/icon.jpg",
    print: false,
    pure: true,
    repo: "bbruceyuan",
    footer: "<script async src=\"//busuanzi.ibruce.info/busuanzi/2.3/busuanzi.pure.mini.js\"></script>本站总访问量<span id=\"busuanzi_value_site_pv\"></span>次,本站访客数<span id=\"busuanzi_value_site_uv\"></span>人次",

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
    hostname: "https://bruceyuan.com",
    favicon: "/img/icon.jpg",

    author: {
        name: "bbruceyuan",
        url: 'https://bruceyuan.com',
    },
    blog: {
        description: "想要开始做一点有意义的事情~",
        medias: {
            Github: "https://github.com/bbruceyuan",
            BiliBili: "https://space.bilibili.com/12420432",
            Twitter: "https://x.com/bbruceyuan"
        },
    },
    plugins: {
        // 配置参考：https://github.com/miniapp-tool/mptool/blob/main/docs/.vuepress/config.ts
        // commid id: da07ca
        // searchPro: {
        //     autoSuggestions: false,
        //     // indexContent: true,
        //     // indexOptions: {
        //     //   tokenize: (text, fieldName) =>
        //     //     fieldName === "id" ? [text] : cut(text, true),
        //     // },
        // },
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
        // redirect: {
        //     config: {
        //         // 年终总结
        //         "/post/15.html": "/blog/2020-year-summary.html",
        //         "/post/34.html": "/blog/2021-year-summary.html",
        //         "/post/35.html": "/blog/2022-year-summary.html",
        //         "/post/2023-year-summary.html": "/blog/2023-year-summary.html",

        //         // 其他
        //         "/post/4.html": "/blog/2020-emnlp-submition.html",
        //         "/post/11.html": "/blog/12.html",
        //         "/post/12.html": "/blog/12.html",
        //         "/post/13.html": "/blog/13.html",
        //         "/post/14.html": "/blog/14.html",
        //         "/post/16.html": "/blog/16.html",
        //         "/post/18.html": "/blog/18.html",
        //         "/post/19.html": "/blog/19.html",
        //         "/post/20.html": "/blog/20.html",
        //         "/post/21.html": "/blog/21.html",
        //         "/post/22.html": "/blog/22.html",
        //         "/post/23.html": "/blog/23.html",
        //         "/post/24.html": "/blog/24.html",
        //         "/post/25.html": "/blog/bai-fei-li-shang-jin-ji.html",
        //         "/post/26.html": "/blog/hub-of-fu-lan-ke-yang.html",
        //         "/post/27.html": "/blog/27.html",
        //         "/post/28.html": "/blog/blind-date-from-bruce.html",
        //         "/post/30.html": "/blog/blind-date-from-miss-cui.html",
        //         "/post/31.html": "/blog/joke-with-miss-cui.html",
        //         "/post/32.html": "/blog/how-i-met-bruce.html",
        //         "/post/33.html": "/blog/life-influenced-by-point.html",
        //         // 40 - 41
        //         "/post/how-to-keep-mental-health-working-in-bytedance.html": "/blog/how-to-keep-mental-health-working-in-bytedance.html",
        //         "/post/ten-years-after-the-college-entrance-examination.html": "/blog/ten-years-after-the-college-entrance-examination.html",
        //     }
        // },
        feed: {
            rss: true,
        },
    },
    headerDepth: 3
})
