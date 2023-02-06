import { defineUserConfig } from 'vuepress'
import { searchPlugin } from "@vuepress/plugin-search";
import { googleAnalyticsPlugin } from '@vuepress/plugin-google-analytics'
import theme from "./theme";


export default defineUserConfig({
    lang: 'zh-CN',
    title: 'BBruceyuan',
    description: '欢迎查看 BBruceyuan 的博客',
    theme,
    plugins: [
        searchPlugin({
          // 你的选项
        }),
        googleAnalyticsPlugin({
          id: 'UA-143053780-1',
        }),
      ],
    markdown: {
      headers: {
        level: [1, 2, 3, 4, 5],
      }
    }
})