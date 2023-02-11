import { defineUserConfig } from 'vuepress'
import { searchPlugin } from "@vuepress/plugin-search";
import { googleAnalyticsPlugin } from '@vuepress/plugin-google-analytics'
import { sitemapPlugin } from "vuepress-plugin-sitemap2";
import theme from "./theme";


export default defineUserConfig({
    lang: 'zh-CN',
    title: 'BBruceyuan',
    description: '欢迎查看 BBruceyuan 的博客',
    head: [
      ['meta', { name: 'baidu-site-verification', content: 'code-8mNFhNa5tZ' }],
      ['meta', { name: '360-site-verification', content: 'b5c713d816b0111fd6e0f0a416d598b3' }],
      ['meta', { name: 'sogou_site_verification', content: 'UBtsNHFicS' }]
    ],
    theme,
    plugins: [
        searchPlugin({
          // 你的选项
        }),
        googleAnalyticsPlugin({
          id: 'G-H2HX76V70M',
        }),
        sitemapPlugin({
          hostname: "https://bbruceyuan.com",
        }),
    ],
    markdown: {
      headers: {
        level: [1, 2, 3, 4, 5],
      }
    }
})