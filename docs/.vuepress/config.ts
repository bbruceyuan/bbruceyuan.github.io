import { viteBundler } from "@vuepress/bundler-vite";
import { googleAnalyticsPlugin } from "@vuepress/plugin-google-analytics";
import { umamiAnalyticsPlugin } from "@vuepress/plugin-umami-analytics";
import { defineUserConfig } from "vuepress";

import theme from "./theme.js";

export default defineUserConfig({
  lang: "zh-CN",
  title: "chaofa用代码打点酱油",
  description:
    "做了一个播客叫做打点酱油，平常写 Python, 对 NLP、计算广告、大模型感兴趣，尝试做一些有意义的事情",

  // 这是我的 一些搜索 相关的配置
  head: [
    // 百度验证
    ["meta", { name: "baidu-site-verification", content: "codeva-y7Qplz9xAV" }],
    // 360 验证
    [
      "meta",
      {
        name: "360-site-verification",
        content: "d65e0e26fb7ffa7c147867834f4d1475",
      },
    ],
    // 搜狗验证
    [
      "meta",
      { "http-equiv": "Content-Type", content: "text/html;charset=gb2312" },
    ],
    ["meta", { name: "sogou_site_verification", content: "sS60nRna6W" }],
    ["meta", { name: "google-adsense-account", content: "ca-pub-6733138658650037"}],
  ],

  markdown: {
    headers: {
      level: [2, 3, 4, 5],
    },
  },

  bundler: viteBundler(),

  theme,

  plugins: [
    googleAnalyticsPlugin({
      id: "G-H2HX76V70M",
    }),
    umamiAnalyticsPlugin({
      id: "e2ad596a-fc3c-4271-9d2c-4be7713aa68f",
      link: "https://ana.bruceyuan.com/script.js",
    }),
  ],
});
