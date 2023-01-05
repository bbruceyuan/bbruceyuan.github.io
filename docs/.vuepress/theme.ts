import { hopeTheme } from "vuepress-theme-hope";
import nav from './nav';
import sidebar from './sidebar';

export default hopeTheme({
    darkmode: "toggle",
    logo: "/img/icon.jpg",
    pure: true,
    editLink: true,
    navbarLayout: {
        left: ["Brand"],
        center: [],
        right: ["Links", "Repo", "Search"]
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
        },
      },
})