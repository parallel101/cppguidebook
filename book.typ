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
    #text(font: "LXGWWenKai", size: 0.9em, fill: rgb("#cd9f0f"))[#body]
]
#let tip = body => box[
    #box(image(
        "pic/bulb.png",
        height: 1em,
    ))
    #text(font: "LXGWWenKai", size: 1em, fill: rgb("#4f8b4f"))[#body]
]
#let warn = body => box[
    #box(image(
        "pic/warning.png",
        height: 1em,
    ))
    #text(font: "LXGWWenKai", size: 1em, fill: rgb("#ed6c6c"))[#body]
]
#let space = block[]

#align(center, text(14pt)[
  *小彭老师的现代 C++ 大典*
])

小彭大典是一本关于现代 C++ 编程的权威指南，它涵盖了从基础知识到高级技巧的内容，适合初学者和有经验的程序员阅读。本书由小彭老师亲自编写，通过简单易懂的语言和丰富的示例，帮助读者快速掌握 C++ 的核心概念，并学会如何运用它们来解决实际问题。

#fun[敢承诺：土木老哥也能看懂！]

= 指南

== 格式约定

这是一段*示例文字*

#tip[用这种颜色字体书写的内容是提示]

#warn[用这种颜色字体书写的内容是警告]

#fun[用这种颜色字体书写的内容是笑话或趣味寓言故事]

/ 术语名称: 这里是术语的定义

== 观前须知

与大多数现有教材不同的是，本课程将会采用“倒叙”的形式，从最新的 *C++23* 讲起！然后讲 C++20、C++17、C++14、C++11，慢慢讲到最原始的 C++98。

不用担心，越是现代的 C++，学起来反而更容易！反而是古代 C++ 又臭又长。

很多同学想当然地误以为 C++98 最简单，哼哧哼哧费老大劲从 C++98 开始学，才是错误的。

为了应付缺胳膊少腿的 C++98，人们发明了各种*繁琐无谓*的写法，在现代 C++ 中，早就已经被更*简洁直观*的写法替代了。

#tip[例如所谓的 safe-bool idiom，写起来又臭又长，C++11 引入一个 `explicit` 关键字直接就秒了。结果还有一批劳保教材大吹特吹 safe-bool idiom，吹得好像是个什么高大上的设计模式一样，不过是个应付 C++98 语言缺陷的蹩脚玩意。]

就好比一个*老外*想要学习汉语，他首先肯定是从*现代汉语*学起！而不是上来就教他*文言文*。

#tip[即使这个老外的职业就是“考古”，或者他对“古代文学”感兴趣，也不可能自学文言文的同时完全跳过现代汉语。]

当我们学习中文时，你肯定希望先学现代汉语，再学文言文，再学甲骨文，再学 brainf\*\*k，而不是反过来。

对于 C++ 初学者也是如此：我们首先学会简单明了的，符合现代人思维的 C++23，再逐渐回到专为伺候“古代开发环境”的 C++98。

你的生产环境可能不允许用上 C++20 甚至 C++23 的新标准。

别担心，小彭老师教会你 C++23 的正常写法后，会讲解如何在 C++14、C++98 中写出同样的效果。

这样你学习的时候思路清晰，不用被繁琐的 C++98 “奇技淫巧”干扰，学起来事半功倍；但也“吃过见过”，知道古代 C++98 的应对策略。

#tip[目前企业里主流使用的是 C++14 和 C++17。例如谷歌就明确规定要求 C++17。]

== 举个例子

#tip[接下来的例子你可能看不懂，但只需要记住这个例子是向你说明：越是新的 C++ 标准，反而越容易学！]

例如，在模板元编程中，要检测一个类型 T 是否拥有 `foo()` 这一成员函数。如果存在，才会调用。

在 C++20 中，可以使用很方便的 `requires` 语法，轻松检测一个表达式是否能合法通过编译。如果能，`requires ` 语句会返回 `true`。然后用一个 `if constexpr` 进行编译期分支判断，即可实现检测到存在则调用。

```cpp
template <class T>
void try_call_foo(T &t) {
    if constexpr (requires { t.foo(); }) {
        t.foo();
    }
}
```

但仅仅是回到 C++17，没有 `requires` 语法，我们只能自己定义一个 trait 类，并运用烦人的 SFINAE 小技巧，检测表达式是否的合法，又臭又长。

```cpp
template <class T, class = void>
struct has_foo {
    inline constexpr bool value = false;
};

template <class T>
struct has_foo<T, std::void_t<decltype(std::declval<T>().foo())>> {
    inline constexpr bool value = true;
};

template <class T>
void try_call_foo(T &t) {
    if constexpr (has_foo<T>::value) {
        t.foo();
    }
}
```

如果回到 C++14，情况就更糟糕了！`if constexpr` 是 C++17 的特性，没有他，要实现编译期分支，我们就得用 `enable_if_t` 的 SFINAE 小技巧，需要定义两个 try_call_foo 函数，互相重载，才能实现同样的效果。

```cpp
template <class T, class = void>
struct has_foo {
    static constexpr bool value = false;
};

template <class T>
struct has_foo<T, std::void_t<decltype(std::declval<T>().foo())>> {
    static constexpr bool value = true;
};

template <class T, std::enable_if_t<has_foo<T>::value, int> = 0>
void try_call_foo(T &t) {
    t.foo();
}

template <class T, std::enable_if_t<!has_foo<T>::value, int> = 0>
void try_call_foo(T &) {
}
```

如果回到 C++11，情况进一步恶化！`enable_if_t` 这个方便的小助手已经不存在，需要使用比他更底层的 `enable_if` 模板类，手动取出 `::type`，并且需要 `typename` 修饰，才能编译通过！并且 `void_t` 也不能用了，要用逗号表达式小技巧才能让 decltype 固定返回 void……

```cpp
template <class T, class = void>
struct has_foo {
    static constexpr bool value = false;
};

template <class T>
struct has_foo<T, decltype(std::declval<T>().foo(), (void)0)> {
    static constexpr bool value = true;
};

template <class T, typename std::enable_if<has_foo<T>::value, int>::type = 0>
void try_call_foo(T &t) {
    t.foo();
}

template <class T, typename std::enable_if<!has_foo<T>::value, int>::type = 0>
void try_call_foo(T &) {
}
```

如果回到 C++98，那又要罪加一等！`enable_if` 和 是 C++11 引入的 `<type_traits>` 头文件的帮手类，在 C++98 中，我们需要自己实现 `enable_if`…… `declval` 也是 C++11 引入的 `<utility>` 头文件中的帮手函数……假设你自己好不容易实现出来了 `enable_if` 和 `declval`，还没完：因为 constexpr 在 C++98 中也不存在了！你无法定义 value 成员变量为编译期常量，我们只好又用一个抽象的枚举小技巧来实现定义类成员常量的效果。

```cpp
template <class T, class = void>
struct has_foo {
    enum { value = 0 };
};

template <class T>
struct has_foo<T, decltype(my_declval<T>().foo(), (void)0)> {
    enum { value = 1 };
};

template <class T, typename my_enable_if<has_foo<T>::value, int>::type = 0>
void try_call_foo(T &t) {
    t.foo();
}

template <class T, typename my_enable_if<!has_foo<T>::value, int>::type = 0>
void try_call_foo(T &) {
}
```

如此冗长难懂的抽象 C++98 代码，仿佛是“加密”过的代码一样，仅仅是为了实现检测是否存在成员函数 foo……

#fun[如果回到 C 语言，那么你甚至都不用检测了。因为伟大的 C 语言连成员函数都没有，何谈“检测成员函数是否存在”？]

反观 C++20 的写法，一眼就看明白代码的逻辑是什么，表达你该表达的，而不是迷失于伺候各种语言缺陷，干扰我们学习。

```cpp
void try_call_foo(auto &t) {
    if constexpr (requires { t.foo(); }) {
        t.foo();
    }
}
```

// 从残废的 C++98 学起，你的思维就被这些无谓的“奇技淫巧”扭曲了，而使得真正应该表达的代码逻辑，淹没在又臭又长的古代技巧中。
// 从现代的 C++23 学起，先知道正常的写法“理应”是什么样。工作中用不上 C++23？我会向你介绍，如果要倒退回 C++14，古代人都是用什么“奇技淫巧”实现同样的效果。
// 这样你最后同样可以适应公司要求的 C++14 环境。但是从 C++23 学起，你的思维又不会被应付古代语言缺陷的“奇技淫巧”扰乱，学起来就事半功倍。

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

TODO: 介绍控制台

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
