// https://www.fonts.net.cn/fonts-zh/tag-kaiti-1.html

#set text(
  font: "Noto Serif CJK SC",
  size: 7pt,
)
#set page(
  paper: "a6",
  margin: (x: 1.8cm, y: 1.5cm),
  header: align(right, text(5pt)[
    小彭大典
  ]),
  numbering: "1",
)
#set par(justify: true)
#set heading(numbering: "1.")
#show "小彭大典": name => box[
    #text(font: "Arial")[✝️]小彭大典#text(font: "Arial")[✝️]
]
#let fun = body => box[
    #box(image(
        "pic/awesomeface.png",
        height: 1em,
    ))
    #text(font: "Aa扁雅楷", size: 0.9em, fill: rgb("#cd9f0f"))[#body]
]
#let tip = body => box[
    #box(image(
        "pic/bulb.png",
        height: 1em,
    ))
    #text(font: "Aa扁雅楷", size: 1em, fill: rgb("#4f8b4f"))[#body]
]
#let warn = body => box[
    #box(image(
        "pic/warning.png",
        height: 1em,
    ))
    #text(font: "Aa扁雅楷", size: 1em, fill: rgb("#ed6c6c"))[#body]
]
#let space = block[]

#align(center, text(14pt)[
  *小彭老师的现代 C++ 大典*
])

小彭大典是一本关于现代 C++ 编程的权威指南，它涵盖了从基础知识到高级技巧的内容，适合初学者和有经验的程序员阅读。本书由小彭老师亲自编写，通过简单易懂的语言和丰富的示例，帮助读者快速掌握 C++ 的核心概念，并学会如何运用它们来解决实际问题。

#fun[敢承诺：土木老哥也能看懂！]

= 指南

== 格式约定

这是一段示例文字

#tip[用这种颜色字体书写的内容是提示]

#warn[用这种颜色字体书写的内容是警告]

#fun[用这种颜色字体书写的内容是笑话或趣味寓言故事]

/ 术语: 这是术语的定义

- 首先
- 其次
- 然后
- 最后

```cpp
// 这是一段示例代码

template <class T>
decltype(T().foo(), std::true_type{}) has_foo(int);

template <class T>
std::false_type has_foo(...);

if constexpr (decltype(has_foo<T>(0))::value) {
    T().foo();
} else {
    otherwise();
}
```

= 开始

== 什么是函数

/ 函数: 一段用 `{}` 包裹的代码块，有一个独一无二的名字做标识。函数可以被其他函数调用。函数可以有返回值和参数。函数的 `{}` 代码块内的程序代码，每次该函数被调用时都会执行。

```cpp
int compute()
{
    return 42;
}
```

上面的代码中，`compute` 就是函数的名字，`int` 表示函数的返回类型——整数。

#tip[乃取整数之英文#quote[integer]的#quote[int]而得名]

而 `{}` 包裹的是函数体，是函数被调用时会执行的代码。

此处 `return 42` 就是函数体内的唯一一条语句，表示函数立即执行完毕，返回 42。

/ 返回值: 当一个函数执行完毕时，会向调用该函数的调用者返回一个值，这个值就是 `return` 后面的表达式的值。返回值可以有不同的类型，此处 `compute` 的返回类型是 `int`，也就是说 `compute` 需要返回一个整数。

#tip[关于函数的参数我们稍后再做说明。]

== 从 main 函数说起

C++ 程序通常由一系列函数组成，其中必须有一个名为 `main` 的函数作为程序的入口点。

main 函数的定义如下：

```cpp
int main()
{
}
```

程序启动时，操作系统会调用 `main` 函数。

#tip[严格来说，是 C++ 运行时调用了 `main` 函数，但目前先理解为#quote[操作系统调用了 `main` 函数]也无妨。]

要把程序发展壮大，我们可以让 `main` 函数调用其他函数，也可以直接在 `main` 函数中编写整个程序的逻辑（不推荐）。

#fun[因此，`main` 可以被看作是#quote[宇宙大爆炸]。]

== main 函数返回值

```cpp
int main()
{
    return 0;
}
```

`return` 表示函数的返回，main 函数返回，即意味着程序的结束。

main 函数总是返回一个整数 (`int` 类型)，用这个整数向操作系统表示程序退出的原因。

如果程序正常执行完毕，正常结束退出，那就请返回 0。

返回一个不为 0 的整数可以表示程序出现了异常，是因为出错了才退出的，值的多少可以用于表明错误的具体原因。

#fun[
    操作系统：我调用了你这个程序的 main 函数，我好奇程序是否正确执行了？让我们约定好：如果你运转正常的话，就返回0表示成功哦！如果有错误的话，就返回一个错误代码，比如返回1表示无权限，2表示找不到文件……之类的。当然，错误代码都是不为0的。
]

== 黑色的窗口？

== 打印一些信息

```cpp
int main()
{
    std::println("Hello, World!");
}
```

以上代码会在控制台输出 `Hello, World!`。

== 注释

```cpp
int main()
{
    // 小彭老师，请你在这里插入程序的逻辑哦！
}
```

这里的 `//` 是注释，注释会被编译器忽略，通常用于在程序源码中植入描述性的文本。有时也会用于多人协作项目中程序员之间互相沟通。

例如下面这段代码：

```cpp
int main()
{
    std::println("编译器伟大，无需多言");
    // 编译器是煞笔
    // 编译器是煞笔
    // 编译器是煞笔
    // 诶嘿你看不见我
}
```

在编译器看来就只是：

```cpp
int main()
{
    std::println("编译器伟大，无需多言");
}
```

#fun[
(\*编译器脸红中\*)
]

#space

C++ 支持行注释 `// xx` 和块注释 `/* xx */` 两种语法。

```cpp
int main()
{
    // 我是行注释
    /* 我是块注释 */
    /* 块注释
        可以
         有
          很多行 */
    std::println(/* 块注释也可以夹在代码中间 */"你好");
    std::println("世界"); // 行注释只能追加在一行的末尾
    std::println("早安");
}
```

#tip[
    在我们以后的案例代码中，都会像这样注释说明，充当*就地讲解员*的效果。去除这些注释并不影响程序的正常运行，添加文字注释只是小彭老师为了提醒你每一行的代码作用。
]

= 函数

函数可以没有返回值，只需要返回类型写 `void` 即可，这样的函数调用的目的只是为了他的副作用（如修改全局变量，输出文本到控制台，修改引用参数等）。

```cpp
void compute()
{
    return;
}
```

#tip[对于没有返回值（返回类型为 `void`）的函数，可以省略 `return`。]

#warn[对于有返回值的函数，必须写 return 语句，否则程序出错。]
