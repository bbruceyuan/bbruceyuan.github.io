import{_ as t,r as p,o,c,b as n,d as e,e as s,a as i}from"./app.984096d0.js";const l="/blog_imgs/10/10_1.png",u={},r=n("h1",{id:"\u84C4\u6C34\u6C60\u7B97\u6CD5",tabindex:"-1"},[n("a",{class:"header-anchor",href:"#\u84C4\u6C34\u6C60\u7B97\u6CD5","aria-hidden":"true"},"#"),s(" \u84C4\u6C34\u6C60\u7B97\u6CD5")],-1),d=n("p",null,"\u84C4\u6C34\u6C60\u7B97\u6CD5\u91C7\u6837\u7B97\u6CD5\u96BE\u7684\u70B9\u4E0D\u5728\u4E8E\u600E\u4E48\u5B9E\u73B0\u84C4\u6C34\u6C60\u7B97\u6CD5\uFF0C\u96BE\u7684\u70B9\u5728\u4E8E\u8BC1\u660E\u6BCF\u4E00\u4E2A\u70B9\u90FD\u80FD\u88AB\u540C\u7B49\u7684\u6982\u7387\u62BD\u53D6\u3002",-1),k=n("p",null,"\u84C4\u6C34\u6C60\u7B97\u6CD5\u8BC1\u660E\u6700\u597D\u7684\u4E24\u4E2A\u6559\u7A0B\u662F\uFF1A",-1),m={href:"https://www.cnblogs.com/snowInPluto/p/5996269.html",target:"_blank",rel:"noopener noreferrer"},v=s("https://www.cnblogs.com/snowInPluto/p/5996269.html"),_={href:"https://www.jianshu.com/p/7a9ea6ece2af",target:"_blank",rel:"noopener noreferrer"},b=s("https://www.jianshu.com/p/7a9ea6ece2af"),h=i('<p>\u5176\u4ED6\u7684\u611F\u89C9\u8BF4\u7684\u4E0D\u662F\u5F88\u6E05\u695A\u3002</p><p>\u84C4\u6C34\u6C60\u7B97\u6CD5\u7684\u4E3B\u8981\u903B\u8F91\uFF1A <img src="'+l+`" alt="image.png"></p><p>\u84C4\u6C34\u6C60\u7B97\u6CD5\u7684Pthon\u5B9E\u73B0\uFF1A</p><div class="language-python ext-py line-numbers-mode"><pre class="language-python"><code><span class="token keyword">import</span> random

<span class="token keyword">def</span> <span class="token function">reservior_sampling</span><span class="token punctuation">(</span>n<span class="token punctuation">,</span> k<span class="token punctuation">)</span><span class="token punctuation">:</span>
    <span class="token triple-quoted-string string">&quot;&quot;&quot;
    \u8868\u793A\u6709 n \u4E2A\u6570\uFF0C\u968F\u673A\u91C7\u6837 k \u4E2A
    &quot;&quot;&quot;</span>
    nums <span class="token operator">=</span> <span class="token punctuation">[</span>i <span class="token keyword">for</span> i <span class="token keyword">in</span> <span class="token builtin">range</span><span class="token punctuation">(</span><span class="token number">1</span><span class="token punctuation">,</span> n <span class="token operator">+</span> <span class="token number">1</span><span class="token punctuation">)</span><span class="token punctuation">]</span>

    res <span class="token operator">=</span> <span class="token punctuation">[</span><span class="token punctuation">]</span>
    <span class="token keyword">for</span> i <span class="token keyword">in</span> <span class="token builtin">range</span><span class="token punctuation">(</span>k<span class="token punctuation">)</span><span class="token punctuation">:</span>
        <span class="token comment"># \u524DK\u4E2A\u6570\u5B57\u53EF\u4EE5\u76F4\u63A5\u586B\u5145</span>
        res<span class="token punctuation">.</span>append<span class="token punctuation">(</span>nums<span class="token punctuation">[</span>i<span class="token punctuation">]</span><span class="token punctuation">)</span>
    
    <span class="token keyword">for</span> i <span class="token keyword">in</span> <span class="token builtin">range</span><span class="token punctuation">(</span>k<span class="token punctuation">,</span> <span class="token builtin">len</span><span class="token punctuation">(</span>nums<span class="token punctuation">)</span><span class="token punctuation">)</span><span class="token punctuation">:</span>
        <span class="token comment"># \u5047\u8BBE i == k (\u4E5F\u662F\u8BF4\u8FD9\u662F \u7B2C k + 1\u4E2A\u5143\u7D20), \u90A3\u4E48 \u8BE5\u6570\u5B57\u6709  k / (k + 1) \u7684\u6982\u7387\u88AB\u9009\u4E2D\u53BB\u53BB\u6362</span>
        replace_idx <span class="token operator">=</span> random<span class="token punctuation">.</span>randint<span class="token punctuation">(</span><span class="token number">0</span><span class="token punctuation">,</span> i<span class="token punctuation">)</span>
        <span class="token keyword">if</span> replace_idx <span class="token operator">&lt;</span> k<span class="token punctuation">:</span>
            res<span class="token punctuation">[</span>replace_idx<span class="token punctuation">]</span> <span class="token operator">=</span> nums<span class="token punctuation">[</span>i<span class="token punctuation">]</span>
    <span class="token keyword">return</span> res

pool <span class="token operator">=</span> reservior_sampling<span class="token punctuation">(</span><span class="token number">100</span><span class="token punctuation">,</span> <span class="token number">10</span><span class="token punctuation">)</span>
<span class="token keyword">print</span><span class="token punctuation">(</span>pool<span class="token punctuation">)</span>
<span class="token comment"># [78, 52, 41, 84, 66, 43, 25, 71, 45, 24]</span>
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div>`,4);function w(g,f){const a=p("ExternalLinkIcon");return o(),c("div",null,[r,d,k,n("ol",null,[n("li",null,[n("a",m,[v,e(a)])]),n("li",null,[n("a",_,[b,e(a)])])]),h])}const x=t(u,[["render",w],["__file","10.html.vue"]]);export{x as default};