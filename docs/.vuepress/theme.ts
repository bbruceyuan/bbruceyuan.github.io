import { hopeTheme } from "vuepress-theme-hope";

import navbar from "./navbar.js";
import sidebar from "./sidebar.js";

export default hopeTheme({
  // basic info
  hostname: "https://yuanchaofa.com",
  favicon: "/img/icon.webp",
  author: {
    name: "Chaofa Yuan",
    url: "https://yuanchaofa.com",
  },

  logo: "/img/icon.webp",
  repo: "bbruceyuan",
  docsRepo: "bbruceyuan/bbruceyuan.github.io",
  docsDir: "docs",

  navbar,
  sidebar,
  footer:
    '<script async src="//busuanzi.ibruce.info/busuanzi/2.3/busuanzi.pure.mini.js"></script>本站总访问量<span id="busuanzi_value_site_pv"></span>次,本站访客数<span id="busuanzi_value_site_uv"></span>人次',

  blog: {
    description: "想要开始做一点有意义的事情，来和我聊天吧~ 个人微信: bbruceyuan",
    medias: {
      Github: "https://github.com/bbruceyuan",
      BiliBili: "https://space.bilibili.com/12420432",
      Youtube: "https://www.youtube.com/@bbruceyuan",
      Twitter: "https://x.com/bbruceyuan",
      XiaoHongShu:
        "https://www.xiaohongshu.com/user/profile/622186a5000000002102bea8",
    },
  },

  // customizations
  navbarLayout: {
    start: ["Brand"],
    center: [],
    end: ["Links", "Repo", "Search", "Outlook"],
  },
  headerDepth: 3,
  contributors: false,
  darkmode: "toggle",
  print: false,
  pure: true,

  hotReload: true,

  plugins: {
    blog: {
      excerpt: false,
    },

    comment: {
      provider: "Giscus",
      repo: "bbruceyuan/bbruceyuan.github.io",
      repoId: "MDEwOlJlcG9zaXRvcnkyNDU3Njc0NzU=",
      category: "Announcements",
      categoryId: "DIC_kwDODqYdM84Cbg9U",
    },

    feed: {
      rss: true,
      count: 10,
    },

    markdownImage: {
      lazyload: true,
    },

    markdownMath: {
      type: "katex",
    },

    mdEnhance: {
      footnote: true,
      tasklist: true,
    },

    searchPro: {
      autoSuggestions: false,
    },
  },
});
