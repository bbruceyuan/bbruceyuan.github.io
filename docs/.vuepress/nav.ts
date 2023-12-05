// .vuepress/navbar.ts
import { navbar } from "vuepress-theme-hope";

export default navbar([
  /* 你的导航栏配置 */
  {
    text: "博客",
    link: "/blog.md"
  },
  {
      text: '友链',
      link: "/link.html",
    },
    
    {
      text: '关于我',
      link: '/about.html',
    },
    {
      text: 'GitHub',
      link: 'https://github.com/bbruceyuan'
    }
]);
