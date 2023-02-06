import { hopeTheme } from "vuepress-theme-hope";
import nav from './nav';
import sidebar from './sidebar';

export default hopeTheme({
    darkmode: "toggle",
    logo: "/img/icon.jpg",
    pure: true,
    // 假如你的文档仓库和项目本身不在一个仓库:
    docsRepo: "bbruceyuan/bbruceyuan.github.io",
    // 假如文档不是放在仓库的根目录下:
    docsDir: "docs",
    // 假如文档放在一个特定的分支下，默认为 'main':
    docsBranch: "source",
    // 默认是 false, 设置为 true 来启用
    
    hotReload: true,
    navbarLayout: {
        start: ["Brand"],
        center: [],
        end: ["Links", "Repo", "Search"]
    },
    navbar: nav,
    sidebar: sidebar,
    blog: {
        description: "这人不想打酱油",
        medias: {
            Github: "https://github.com/bbruceyuan",
            BiliBili: "https://space.bilibili.com/12420432",
            Twitter: "https://twitter.com/BBruceyuan"
        },
    },
    plugins: {
        blog: {
            excerptLength: 0,
            excerpt: false,
        },
        mdEnhance: {
          // 使用 KaTeX 启用 TeX 支持
          katex: true,
          footnote: true,
          tasklist: true,
        },
      },
})