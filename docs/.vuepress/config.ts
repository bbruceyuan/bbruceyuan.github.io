import { defineUserConfig } from 'vuepress'
import { searchPlugin } from "@vuepress/plugin-search";
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
      ],
})