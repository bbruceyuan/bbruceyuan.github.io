import { defineUserConfig } from 'vuepress'
import { mdEnhancePlugin } from "vuepress-plugin-md-enhance";
import { defaultTheme } from 'vuepress'

export default defineUserConfig({
  lang: 'zh-CN',
  title: 'BBruceyuan',
  description: '欢迎查看 BBruceyuan 的博客',
  theme: defaultTheme({
    navbar: [
      {
        text: '归档',
        children: [{text: "随笔", link: '/post/4.html/'}, 
        {text: "技术", link: '/post/1.html/'}],
      },
      {
        text: '友链',
        link: "/link.html/",
      },
      
      {
        text: '关于我',
        link: '/about/',
      },
      {
        text: 'GitHub',
        link: 'https://github.com/bbruceyuan'
      }
    ],
    sidebarDepth: 3,
    sidebar: [
      {
        text: '年终总结',
        children: [
            "/post/年终总结/15.2020年过去了，我不怀念它.md",
            "/post/年终总结/34.2021，乌云与曙光.md",
        ]
      },
      {
        text: '随笔',
        collapsible: false,
        children: [
            '/post/随笔/12.我不喜欢失眠.md',
            '/post/随笔/13.倒着写的文章.md',
            '/post/随笔/14.BBruceyuan的近期不靠谱事件.md',
            '/post/随笔/16.周四，又见周四.md',
            '/post/随笔/18.我从没这么喜欢待在家里.md',
            '/post/随笔/19.我的黄金时代已经过去.md',
            '/post/随笔/20.三月的三分之一.md',
            '/post/随笔/21.翻页了.md',
            '/post/随笔/22.三月末的陈词.md',
            '/post/随笔/23.一个粉丝的自我修养.md',
            '/post/随笔/24.洗澡.md',
            '/post/随笔/25.《白非立上进记合集》.md',
            '/post/随笔/26.《弗兰克扬小说合集》.md',
            '/post/随笔/27.香格里拉封闭培训的七天.md',
            '/post/随笔/28.我没想到我会误入相亲素材.md',
            '/post/随笔/30.崔同学视角的「我没想到我会误入相亲素材」.md',
            '/post/随笔/31.和崔同学日常段子集锦.md',
            {text: 'How-I-Met-Bruce?', link: '/post/32.html/'},
            '/post/随笔/33.Life-Influenced-By-Point.md',
            '/post/随笔/4.记EMNLP2020投稿.md',
        ],
      },
      {
        text: '技术',
        children: [
            '/post/1.2020年了，还有必要学习分词算法吗？.md', 
            '/post/2.深度学习时代，分词算法的真实应用实例.md',
            '/post/3.关于隐马尔可夫模型(HMM)，需要知道什么？.md',
            '/post/5.Transition-based-Directed-Graph-Construction-for-Emotion-Cause-Pair-Extraction(中文介绍).md',
            '/post/7.Must-read-Papers-on-Emotion-Cause-Pair-Extraction.md',
            '/post/8.01之间均匀分区取两点构成三角形的概率-证明加代码实现.md',
            '/post/9.简单方法增加Query召回的多样性.md',
            '/post/10.Python实现蓄水池算法.md',
            '/post/17.NER上分利器-实体边界重定位.md',
            '/post/29.倒排索引原理与python实现.md'
        ],
      },
    ]
  }),
  plugins: [
    mdEnhancePlugin({
      // 启用 TeX 支持
      tex: true,
    }),
  ],
})