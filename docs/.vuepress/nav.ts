// .vuepress/navbar.ts
import { navbar } from "vuepress-theme-hope";

export default navbar([
  /* 你的导航栏配置 */
  {
    text: "计算广告入门",
    link: "/introduction-to-computing-advertising"
  },
  {
    text: "博客",
    link: "/blog.md"
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
]);
