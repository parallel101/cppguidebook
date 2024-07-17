# 小彭老师的现代 C++ 大典

小彭大典是一本关于现代 C++ 编程的权威指南，它涵盖了从基础知识到高级技巧的内容，适合初学者和有经验的程序员阅读。本书由小彭老师亲自编写，通过简单易懂的语言和丰富的示例，帮助读者快速掌握 C++ 的核心概念，并学会如何运用它们来解决实际问题。

> [!NOTE]
> 敢承诺：土木老哥也能看懂！

## 下载 PDF

[点击开始在线阅读](https://142857.red/files/cppguidebook.pdf)

也可以前往 [GitHub Release 页面](https://github.com/parallel101/cppguidebook/releases) 下载 PDF 文件。

> [!NOTE]
> 感谢 [Derived Cat](https://github.com/hooyuser) 大佬提供智能脚本！每当小彭老师推送了修改后，Release 页面和小水管服务器上的 PDF 文件都会自动更新。只要看到小彭老师提交新 commit，你就可以随时重新下载最新版。（如果看到没有变化,是浏览器缓存导致的，建议电脑端按 F5 或者手机屏幕下拉刷新一下）

## 你也可以参与编写的开源小册

本书完全开源，源文件为 [`cppguidebook.typ`](cppguidebook.typ)。

如果发现书写问题，或者你有想加入的新章节，有关于 C++ 新想法，新技巧分享给大家，可以提交 [Pull Request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request) 来帮助小彭老师一起写书。合并后，GitHub 的机器人讲自动重新编译出 PDF。

添加新章节，可以从下面的“大纲 / Roadmap”中挑选，认领其中的一个章节，开始编写。也可以选择一个大纲里没有，但你认为很重要的方面来写。

# 大纲 / Roadmap

- 前言 (完成)
- 开发环境与平台选择 (小彭老师施工中)
- 你好，世界 (小彭老师施工中)
- 变量与类型 (待认领)
- 认识标准库 (待认领)
- 自定义函数 (待认领)
- 自定义类型 (待认领)
- 标准库容器 (待认领)
- 成员函数 (待认领)
- 我们需要多态 (待认领)
- 自定义模板 (待认领)
- TODO: 更多章节

## 赞助名单

感谢以下小彭友的赞助！

[![Thanks list](pic/thanks.png)](https://afdian.net/a/archibate)

> [!NOTE]
> 小彭老师的大典是免费下载的，不用赞助也可以查看哦。

小彭老师遭到 [“白眼狼”脑板](https://zjnews.zjol.com.cn/zjnews/hznews/201612/t20161202_2143682.shtml) 开除，目前处于失业状态。只好寻求各位小彭友赞助，保障小彭老师的基本生命体征运行。

> [!TIP]
> 小彭老师领衔开发的 [Zeno](https://github.com/zenustech/zeno) 软件，曾参与 [流量地球 2](https://t.cj.sina.com.cn/articles/view/1738690784/v67a250e0019013tli)、杭州亚运会等大型项目的特效制作，魅惑无数西装大脑投资人，为“白眼狼”博得风光无限。现在却将如此贡献巨大的 Zeno “开国功勋”，以“资金困难”为由“卸载”了，足以见这位“白眼狼”的“知恩图报”。

如果你觉得本书对你有所帮助，可以通过 [爱发电](https://afdian.net/a/archibate) 赞助小彭老师，以便小彭老师有更多的精力继续编写和维护本书。

> [!TIP]
> 每有一位小彭友赞助 `26.90`，小彭老师一天的食品安全就有了着落。

<a href="https://afdian.net/a/archibate"><img src="https://142857.red/afdian-qrcode.jpg?y" alt="https://afdian.net/a/archibate" width="400px"/></a>

> [!WARN]
> 救命……爱发电似乎关停了！？小彭老师赶紧贴出支付宝收款码作为替代……

<img src="misc/zfb-qrcode.jpg" alt="misc/zfb-qrcode.jpg" width="400px"/>

> [!TIP]
> 如果你也处于失业状态，就不用勉强赞助了……也可以先给小彭老师点一颗⭐Star⭐表示心意。

## Typst 真好用，家人们

本书使用 [Typst 语言](https://github.com/typst/typst) 书写，类似于 LaTeX 或 Markdown。和 Markdown 相比，Typst 的排版功能更加丰富。和 LaTeX 相比，Typst 更加轻量级，语法简单明了。最重要的是，Typst 支持宏编程，可以在文本中书写代码，实现批量生成结构化的文本。

![Typst](https://user-images.githubusercontent.com/17899797/228031796-ced0e452-fcee-4ae9-92da-b9287764ff25.png)

Typst 语言书写的 `.typ` 源文件编译后，得到可供阅读的 `.pdf` 文件。

克隆本仓库后，可以用 `typst compile cppguidebook.typ` 命令编译生成 `cppguidebook.pdf` 文件，也可以用 [`typst-preview`](https://github.com/Enter-tainer/typst-preview`) 等工具，一边修改源文件，一边实时预览效果。

> 以下是第一章节的内容预览，要查看全文，请前往 Release 页面下载完整 PDF 文件。

# 前言

推荐用手机或平板**竖屏**观看，可以在床或沙发上躺着。

用电脑看的话，可以按 `WIN + ←`，把本书的浏览器窗口放在屏幕左侧，右侧是你的 IDE。一边看一边自己动手做实验。

![split view](pic/slide.jpg)

> 请坐和放宽。

## 观前须知

与大多数现有教材不同的是，本课程将会采用“倒叙”的形式，从最新的 **C++23** 讲起！然后讲 C++20、C++17、C++14、C++11，慢慢讲到最原始的 C++98。

不用担心，越是现代的 C++，学起来反而更容易！反而古代 C++ 才**又臭又长**。

很多同学想当然地误以为 C++98 最简单，哼哧哼哧费老大劲从 C++98 开始学，才是错误的。

为了应付缺胳膊少腿的 C++98，人们发明了各种**繁琐无谓**的写法，在现代 C++ 中，早就已经被更**简洁直观**的写法替代了。

> [!TIP]
> 例如所谓的 safe-bool idiom，写起来又臭又长，C++11 引入一个 `explicit` 关键字直接就秒了。结果还有一批劳保教材大吹特吹 safe-bool idiom，吹得好像是个什么高大上的设计模式一样，不过是个应付 C++98 语言缺陷的蹩脚玩意。

就好比一个**老外**想要学习汉语，他首先肯定是从**现代汉语**学起！而不是上来就教他**文言文**。

> [!TIP]
> 即使这个老外的职业就是“考古”，或者他对“古代文学”感兴趣，也不可能自学文言文的同时完全跳过现代汉语。

当我们学习中文时，你肯定希望先学现代汉语，再学文言文，再学甲骨文，再学 brainf\*\*k，而不是反过来。

对于 C++ 初学者也是如此：我们首先学会简单明了的，符合现代人思维的 C++23，再逐渐回到专为伺候“古代开发环境”的 C++98。

你的生产环境可能不允许用上 C++20 甚至 C++23 的新标准。

别担心，小彭老师教会你 C++23 的正常写法后，会讲解如何在 C++14、C++98 中写出同样的效果。

这样你学习的时候思路清晰，不用被繁琐的 C++98 “奇技淫巧”干扰，学起来事半功倍；但也“吃过见过”，知道古代 C++98 的应对策略。

> [!TIP]
> 目前企业里主流使用的是 C++14 和 C++17。例如谷歌就明确规定要求 C++17。

# 开发环境与平台选择

[>> 继续阅读剩余章节](https://142857.red/files/cppguidebook.pdf)
