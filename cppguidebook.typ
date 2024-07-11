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
    #text(font: "LXGWWenKai", size: 1em, fill: rgb("#cd9f0f"))[#body]
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
#let story = body => box[
    #box(image(
        "pic/book.png",
        height: 1em,
    ))
    #text(font: "LXGWWenKai", size: 1em, fill: rgb("#807340"))[#body]
]
#let detail = body => box[
    #box(image(
        "pic/question.png",
        height: 1em,
    ))
    #text(font: "LXGWWenKai", size: 1em, fill: rgb("#8080ad"))[#body]
]
#let space = block[]

#align(center, text(14pt)[
  *小彭老师的现代 C++ 大典*
])

小彭大典是一本关于现代 C++ 编程的权威指南，它涵盖了从基础知识到高级技巧的内容，适合初学者和有经验的程序员阅读。本书由小彭老师亲自编写，通过简单易懂的语言和丰富的示例，帮助读者快速掌握 C++ 的核心概念，并学会如何运用它们来解决实际问题。

#fun[敢承诺：土木老哥也能看懂！]

= 前言

推荐用手机或平板*竖屏*观看，可以在床或沙发上躺着。

用电脑看的话，可以按 `WIN + ←`，把本书的浏览器窗口放在屏幕左侧，右侧是你的 IDE。一边看一边自己动手做实验。

#image("pic/slide.jpg")

#fun[请坐和放宽。]

== 格式约定

#tip[用这种颜色字体书写的内容是温馨提示]

#warn[用这种颜色字体书写的内容是可能犯错的警告]

#fun[用这种颜色字体书写的内容是笑话或趣味寓言故事]

#story[用这种颜色书写的是补充说明的课外阅读，看不懂也没关系]

#detail[用这种颜色字体书写的是初学者可暂时不用理解的细节]

/ 术语名称: 这里是术语的定义。

== 观前须知

与大多数现有教材不同的是，本课程将会采用“倒叙”的形式，从最新的 *C++23* 讲起！然后讲 C++20、C++17、C++14、C++11，慢慢讲到最原始的 C++98。

不用担心，越是现代的 C++，学起来反而更容易！反而古代 C++ 才*又臭又长*。

很多同学想当然地误以为 C++98 最简单，哼哧哼哧费老大劲从 C++98 开始学，才是错误的。

为了应付缺胳膊少腿的 C++98，人们发明了各种*繁琐无谓*的写法，在现代 C++ 中，早就已经被更*简洁直观*的写法替代了。

#story[例如所谓的 safe-bool idiom，写起来又臭又长，C++11 引入一个 `explicit` 关键字直接就秒了。结果还有一批劳保教材大吹特吹 safe-bool idiom，吹得好像是个什么高大上的设计模式一样，不过是个应付 C++98 语言缺陷的蹩脚玩意。]

就好比一个*老外*想要学习汉语，他首先肯定是从*现代汉语*学起！而不是上来就教他*文言文*。

#fun[即使这个老外的职业就是“考古”，或者他对“古代文学”感兴趣，也不可能自学文言文的同时完全跳过现代汉语。]

当我们学习中文时，你肯定希望先学现代汉语，再学文言文，再学甲骨文，再学 brainf\*\*k，而不是反过来。

对于 C++ 初学者也是如此：我们首先学会简单明了的，符合现代人思维的 C++23，再逐渐回到专为伺候“古代开发环境”的 C++98。

你的生产环境可能不允许用上 C++20 甚至 C++23 的新标准。

别担心，小彭老师教会你 C++23 的正常写法后，会讲解如何在 C++14、C++98 中写出同样的效果。

这样你学习的时候思路清晰，不用被繁琐的 C++98 “奇技淫巧”干扰，学起来事半功倍；但也“吃过见过”，知道古代 C++98 的应对策略。

#tip[目前企业里主流使用的是 C++14 和 C++17。例如谷歌就明确规定要求 C++17。]

== 举个例子

#story[接下来的例子你可能看不懂，但只需要记住这个例子是向你说明：越是新的 C++ 标准，反而越容易学！]

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

= 开发环境与平台选择

TODO

== IDE 不是编译器！

TODO

== 编译器是？

编译器是将源代码 (`.cpp`) 编译成可执行程序 (`.exe`) 的工具。

#fun[C++ 是*编译型语言*，源代码不能直接执行哦！刚开始学编程的小彭老师曾经把网上的 “Hello, World” 代码拷贝到 `.c` 源码文件中，然后把后缀名改成 `.exe`，发现这样根本执行不了……后来才知道需要通过一种叫做*编译器*编译 `.c` 文件，才能得到计算机可以直接执行的 `.exe` 文件。]

C++ 源码 `.cpp` 是写给人类看的！计算机并不认识，计算机只认识二进制的机器码。要把 C++ 源码转换为计算机可以执行的机器码。

== 编译器御三家

最常见的编译器有：GCC、Clang、MSVC

#fun[俗称“御三家”。]

这些编译器都支持了大部分 C++20 标准和小部分 C++23 标准，而 C++17 标准都是完全支持的。

#fun[有人说过：“如果你不知道一个人是用的什么编译器，那么你可以猜他用的是 GCC。”]

- GCC 主要只在 Linux 和 MacOS 等 Unix 类系统可用，不支持 Windows 系统。但是 GCC 有着大量好用的扩展功能，例如大名鼎鼎的 `pbds`（基于策略的数据结构），还有各种 `__attribute__`，各种 `__builtin_` 系列函数。不过随着新标准的出台，很多原本属于 GCC 的功能都成了标准的一部分，例如 `__attribute__((warn_unused))` 变成了标准的 `[[nodiscard]]`，`__builtin_clz` 变成了标准的 `std::countl_zero`，`__VA_OPT__` 名字都没变就进了 C++20 标准。

#fun[PBDS 又称 “平板电视”]

- 也有 MinGW 这样的魔改版 GCC 编译器，把 GCC 移植到了 Windows 系统上，同时也能用 GCC 的一些特性。不过 MinGW 最近已经停止更新，最新的 GCC Windows 移植版由 MinGW-w64 继续维护。

- Clang 是跨平台的编译器，支持大多数主流平台，包括操作系统界的御三家：Linux、MacOS、Windows。Clang 支持了很大一部分 GCC 特性和部分 MSVC 特性。其所属的 LLVM 项目更是编译器领域的中流砥柱，不仅支持 C、C++、Objective-C、Fortran 等，Rust 和 Swift 等语言也是基于 LLVM 后端编译的，不仅如此，还有很多显卡厂商的 OpenGL 驱动也是基于 LLVM 实现编译的。并且 Clang 身兼数职，不仅可以编译，还支持静态分析。许多 IDE 常见的语言服务协议 (LSP) 就是基于 Clang 的服务版————Clangd 实现的 (例如你可以按 Ctrl 点击，跳转到函数定义，这样的功能就是 IDE 通过调用 Clangd 的 LSP 接口实现）。不过 Clang 的性能优化比较激进，虽然有助于性能提升，如果你不小心犯了未定义行为，Clang 可能优化出匪夷所思的结果，如果你要实验未定义行为，Clang 是最擅长复现的。且 Clang 对一些 C++ 新标准特性支持相对较慢，没有 GCC 和 MSVC 那么上心。

#story[例如 C++20 早已允许 lambda 表达式捕获 structural-binding 变量，而 Clang 至今还没有支持，尽管 Clang 已经支持了很多其他 C++20 特性。]

- Apple Clang 是苹果公司自己魔改的 Clang 版本，只在 MacOS 系统上可用，支持 Objective-C 和 Swift 语言。但是版本较官方 Clang 落后一些，很多新特性都没有跟进，基本上只有专门伺候苹果的开发者会用。

#tip[GCC 和 Clang 也支持 Objective-C。]

- MSVC 是 Windows 限定的编译器，提供了很多 MSVC 特有的扩展。也有人在 Clang 上魔改出了 MSVC 兼容模式，兼顾 Clang 特性的同时，支持了 MSVC 的一些特性（例如 `__declspec`），可以编译用了 MSVC 特性的代码，即 `clang-cl`，在最新的 VS2022 IDE 中也集成了 `clang-cl`。值得注意的是，MSVC 的优化能力是比较差的，比 GCC 和 Clang 都差，例如 MSVC 几乎总是假定所有指针 aliasing，这意味着当遇到很多指针操作的循环时，几乎没法做循环矢量化。但是也使得未定义行为不容易产生 Bug，另一方面，这也导致一些只用 MSVC 的人不知道某些写法是未定义行为。

- Intel C++ compiler 是英特尔开发的 C++ 编译器，由于是硬件厂商开发的，特别擅长做性能优化。但由于更新较慢，基本没有更上新特性，也没什么人在用了。

#detail[最近他们又出了个 Intel DPC++ compiler，支持最新的并行编程领域特定语言 SyCL。]

== 使用编译器编译源码

=== MSVC

```cmd
cl.exe /c main.cpp 
```

这样就可以得到可执行文件 `main.exe` 了。

=== GCC

```bash
g++ -c main.cpp -o main
```

这样就可以得到可执行文件 `main` 了。

#tip[Linux 系统的可执行文件并没有后缀名，所以没有 `.exe` 后缀。]

=== Clang

Windows 上：

```bash
clang++.exe -c main.cpp -o main.exe
```

Linux / MacOS 上：

```bash
clang++ -c main.cpp -o main
```

== 编译器选项

编译器选项是用来控制编译器的行为的。不同的编译器有不同的选项，语法有微妙的不同，但大致功效相同。

例如当我们说“编译这个源码时，我用了 GCC 编译器，`-O3` 和 `-std=c++20` 选项”，说的就是把这些选项加到了 `g++` 的命令行参数中：

```bash
g++ -O3 -std=c++20 -c main.cpp -o main
```

其中 Clang 和 GCC 的编译器选项有很大交集。而 MSVC 基本自成一派。

Clang 和 GCC 的选项都是 `-xxx` 的形式，MSVC 的选项是 `/xxx` 的形式。

常见的编译器选项有：

=== C++ 标准

指定要选用的 C++ 标准。

Clang 和 GCC：`-std=c++98`、`-std=c++03`、`-std=c++11`、`-std=c++14`、`-std=c++17`、`-std=c++20`、`-std=c++23`

MSVC：`/std:c++98`、`/std:c++11`、`/std:c++14`、`/std:c++17`、`/std:c++20`、`/std:c++latest`

例如要编译一个 C++20 源码文件，分别用 GCC、Clang、MSVC：

GCC（Linux）：

```bash
g++ -std=c++20 -c main.cpp -o main
```

Clang（Linux）：

```bash
clang++ -std=c++20 -c main.cpp -o main
```

MSVC（Windows）：

```bash
cl.exe /std:c++20 /c main.cpp
```

=== 优化等级

Clang 和 GCC：`-O0`、`-O1`、`-O2`、`-O3`、`-Ofast`、`-Os`、`-Oz`、`-Og`

- `-O0`：不进行任何优化，编译速度最快，忠实复刻你写的代码，未定义行为不容易产生诡异的结果，一般用于开发人员内部调试阶段。
- `-O1`：最基本的优化，会把一些简单的死代码（编译器检测到的不可抵达代码）删除，去掉没有用的变量，把部分变量用寄存器代替等，编译速度较快，执行速度也比 `-O0` 快。但是会丢失函数的行号信息，影响诸如 gdb 等调试，如需快速调试可以用 `-Og` 选项。
- `-O2`：比 `-O1` 更强的优化，会把一些循环展开，把一些函数内联，减少函数调用，把一些简单的数组操作用更快的指令替代等，执行速度更快。
- `-O3`：比 `-O2` 更激进的优化，会把一些复杂的循环用 SIMD 矢量指令优化加速，把一些复杂的数组操作用更快的指令替代等。性能提升很大，但是如果你的程序有未定义行为，可能会导致一些 Bug。如果你的代码没有未定义行为则绝不会有问题，对自己的代码质量有自信就可以放心开，编译速度也会很慢，一般用于程序最终成品发布阶段。
- `-Ofast`：在 `-O3` 的基础上，进一步对浮点数的运算进行更深层次的优化，但是可能会导致一些浮点数计算结果不准确。如果你的代码不涉及到 NaN 和 Inf 的处理，那么 `-Ofast` 不会有太大的问题，一般用于科学计算领域的终极性能优化。
- `-Os`：在 `-O2` 的基础上，专门优化代码大小，性能被当作次要需求，但是会禁止会导致可执行文件变大的优化。会把一些循环展开、内联等优化关闭，把一些代码用更小的指令实现，尽可能减小可执行文件的尺寸，比 `-O0`、`-O1`、`-O2` 都要小，通常用于需要节省内存的嵌入式系统开发。
- `-Oz`：在 `-Os` 的基础上，进一步把代码压缩，可能把本可以一条大指令完成的任务也拆成多条小指令，为了尺寸完全性能，大幅减少了函数内联的机会，有时用于嵌入式系统开发。
- `-Og`：在 `-O0` 的基础上，尽可能保留更多调试信息，不做破坏函数行号等信息的优化，建议配合产生更多调试信息的 `-g` 选项使用。但还是会做一些简单的优化，比 `-O0` 执行速度更快。但 `-Og` 的所有优化都不会涉及到未定义行为，因此非常适合调试未定义行为。但是由于插入了调试信息，最终的可执行文件会变得很大，一般在开发人员调试时使用。

MSVC：`/Od`、`/O1`、`/O2`、`/Ox`、`/Ob1`、`/Ob2`、`/Os`

- `/Od`：不进行任何优化，忠实复刻你写的代码，未定义行为不容易产生诡异的结果，一般用于调试阶段。
- `/O1`：最基本的优化，会把一些简单的死代码删除，去掉没有用的变量，把变量用寄存器代替等。
- `/O2`：比 `/O1` 更强的优化，会把一些循环展开，把一些函数内联，减少函数调用，还会尝试把一些循环矢量化，把一些简单的数组操作用更快的指令替代等。一般用于发布阶段。
- `/Ox`：在 `/O2` 的基础上，进一步优化，但是不会导致未定义行为，一般用于发布阶段。
- `/Ob1`：启用函数内联。
- `/Ob2`：启用函数内联，但是会扩大内联范围，一般比 `/Ob1` 更快，但是也会导致可执行文件变大。
- `/Os`：在 `/O2` 的基础上，专门优化代码大小，性能被当作次要需求，但是会禁止会导致可执行文件变大的优化。会把一些循环展开、内联等优化关闭，把一些代码用更小的指令实现，尽可能减小可执行文件的尺寸，通常用于需要节省内存的嵌入式系统开发。

=== 调试信息

Clang 和 GCC：`-g`、`-g0`、`-g1`、`-g2`、`-g3`

MSVC：`/Z7`、`/Zi`

=== 头文件搜索路径

=== 指定要链接的库

=== 库文件搜索路径

=== 定义宏

Clang 和 GCC：`-Dmacro=value`

MSVC：`/Dmacro=value`

例如：

=== 警告开关

== 标准库御三家

- libstdc++ 是 GCC 官方的 C++ 标准库实现，由于 GCC 是 Linux 系统的主流编译器，所以 libstdc++ 也是 Linux 上最常用的标准库。你可以在这里看到他的源码：https://github.com/gcc-mirror/gcc/tree/master/libstdc%2B%2B-v3

- libc++ 是 Clang 官方编写的 C++ 标准库实现，由于 Clang 是 MacOS 系统的主流编译器，所以 libc++ 也是 MacOS 上最常用的标准库。libc++ 也是 C++ 标准库中最早实现 C++11 标准的。项目的开源地址是：https://github.com/llvm/llvm-project/tree/main/libcxx

- MSVC STL 是 MSVC 官方的 C++ 标准库实现，由于 MSVC 是 Windows 系统的主流编译器，所以 MSVC STL 也是 Windows 上最常用的标准库。MSVC STL 也是 C++ 标准库中最晚实现 C++11 标准的，但是现在他已经完全支持 C++20，并且也完全开源了：https://github.com/microsoft/STL

值得注意的是，标准库和编译器并不是绑定的，例如 Clang 可以用 libstdc++ 或 MSVC STL，GCC 也可以被配置使用 libc++。

在 Linux 系统中，Clang 默认用的就是 libstdc++。需要为 Clang 指定 `-stdlib=libc++` 选项，才能使用。

#fun[牛头人笑话：“如果你不知道一个人是用的什么标准库，那么你可以猜他用的是 libstdc++。因为即使他的编译器是 Clang，他用的大概率依然是 libstdc++。”]

=== 标准库的调试模式

TODO

= 你好，世界

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

#detail[严格来说，是 C++ 运行时调用了 `main` 函数，但目前先理解为#quote[操作系统调用了 `main` 函数]也无妨。]

要把程序发展壮大，我们可以让 `main` 函数调用其他函数，也可以直接在 `main` 函数中编写整个程序的逻辑（不推荐）。

#fun[因此，`main` 可以被看作是#quote[宇宙大爆炸]。]

== main 函数的返回值

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

== 这个黑色的窗口是？

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

= 变量与类型

TODO

= 自定义函数

函数可以没有返回值，只需要返回类型写 `void` 即可，这样的函数调用的目的只是为了他的副作用（如修改全局变量，输出文本到控制台，修改引用参数等）。

```cpp
void compute()
{
    return;
}
```

#tip[对于没有返回值（返回类型为 `void`）的函数，可以省略 `return` 不写。]

#warn[对于有返回值的函数，必须写 return 语句，否则程序出错。]

TODO：更多介绍函数

= 函数式编程

== 为什么需要函数？

```cpp
int main() {
    std::vector<int> a = {1, 2, 3, 4};
    int s = 0;
    for (int i = 0; i < a.size(); i++) {
        s += a[i];
    }
    fmt::println("sum = {}", s);
    return 0;
}
```

这是一个计算数组求和的简单程序。

但是，他只能计算数组 a 的求和，无法复用。

如果我们有另一个数组 b 也需要求和的话，就得把整个求和的 for 循环重新写一遍：

```cpp
int main() {
    std::vector<int> a = {1, 2, 3, 4};
    int s = 0;
    for (int i = 0; i < a.size(); i++) {
        s += a[i];
    }
    fmt::println("sum of a = {}", s);
    std::vector<int> b = {5, 6, 7, 8};
    s = 0;
    for (int i = 0; i < a.size(); i++) {
        s += b[i];
    }
    fmt::println("sum of b = {}", s);
    return 0;
}
```

这就出现了程序设计的大忌：代码重复。

#fun[例如，你有吹空调的需求，和充手机的需求。你为了满足这两个需求，购买了两台发电机，分别为空调和手机供电。第二天，你又产生了玩电脑需求，于是你又购买一台发电机，专为电脑供电……真是浪费！]

重复的代码不仅影响代码的*可读性*，也增加了*维护*代码的成本。

+ 看起来乱糟糟的，信息密度低，让人一眼看不出代码在干什么的功能
+ 很容易写错，看走眼，难调试
+ 复制粘贴过程中，容易漏改，比如这里的 `s += b[i]` 可能写成 `s += a[i]` 而自己不发现
+ 改起来不方便，当我们的需求变更时，需要多处修改，比如当我需要改为计算乘积时，需要把两个地方都改成 `s *=`
+ 改了以后可能漏改一部分，留下 Bug 隐患
+ 敏捷开发需要反复修改代码，比如你正在调试 `+=` 和 `-=` 的区别，看结果变化，如果一次切换需要改多处，就影响了调试速度

=== 狂想：没有函数的世界？

#story[如果你还是喜欢“一本道”写法的话，不妨想想看，完全不用任何标准库和第三方库的函数和类，把 `fmt::println` 和 `std::vector` 这些函数全部拆解成一个个系统调用。那这整个程序会有多难写？]

```cpp
int main() {
#ifdef _WIN32
    int *a = (int *)VirtualAlloc(NULL, 4096, MEM_COMMIT, PAGE_EXECUTE_READWRITE);
#else
    int *a = (int *)mmap(NULL, 4 * sizeof(int), PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
#endif
    a[0] = 1;
    a[1] = 2;
    a[2] = 3;
    a[3] = 4;
    int s = 0;
    for (int i = 0; i < 4; i++) {
        s += a[i];
    }
    char buffer[64];
    buffer[0] = 's';
    buffer[1] = 'u';
    buffer[2] = 'm';
    buffer[3] = ' ';
    buffer[4] = '=';
    buffer[5] = ' '; // 例如，如果要修改此处的提示文本，甚至需要修改后面的 len 变量...
    int len = 6;
    int x = s;
    do {
        buffer[len++] = '0' + x % 10;
        x /= 10;
    } while (x);
    buffer[len++] = '\n';
#ifdef _WIN32
    WriteFile(GetStdHandle(STD_OUTPUT_HANDLE), buffer, len, NULL, NULL);
#else
    write(1, buffer, len);
#endif
    int *b = (int *)a;
    b[0] = 4;
    b[1] = 5;
    b[2] = 6;
    b[3] = 7;
    int s = 0;
    for (int i = 0; i < 4; i++) {
        s += b[i];
    }
    len = 6;
    x = s;
    do {
        buffer[len++] = '0' + x % 10;
        x /= 10;
    } while (x);
    buffer[len++] = '\n';
#ifdef _WIN32
    WriteFile(GetStdHandle(STD_OUTPUT_HANDLE), buffer, len, NULL, NULL);
#else
    write(1, buffer, len);
#endif
#ifdef _WIN32
    VirtualFree(a, 0, MEM_RELEASE);
#else
    munmap(a);
#endif
    return 0;
}
```

不仅完全没有可读性、可维护性，甚至都没有可移植性。

除非你只写应付导师的“一次性”程序，一旦要实现复杂的业务需求，不可避免的要自己封装函数或类。网上所有鼓吹“不封装”“设计模式是面子工程”的反智言论，都是没有做过大型项目的。


=== 设计模式追求的是“可改”而不是“可读”！

很多设计模式教材片面强调*可读性*，仿佛设计模式就是为了“优雅”“高大上”“美学”？使得很多人认为，“我这个是自己的项目，不用美化给领导看”而拒绝设计模式。实际上设计模式的主要价值在于*方便后续修改*！

#fun[例如 B 站以前只支持上传普通视频，现在叔叔突然提出：要支持互动视频，充电视频，视频合集，还废除了视频分 p，还要支持上传短视频，竖屏开关等……每一个叔叔的要求，都需要大量程序员修改代码，无论涉及前端还是后端。]

与建筑、绘画等领域不同，一次交付完毕就可以几乎永久使用。而软件开发是一个持续的过程，每次需求变更，都导致代码需要修改。开发人员几乎需要一直围绕着软件代码，不断的修改。调查表明，程序员 90% 的时间花在*改代码*上，*写代码*只占 10%。

#fun[软件就像生物，要不断进化，软件不更新不维护了等于死。如果一个软件逐渐变得臃肿难以修改，无法适应新需求，那他就像已经失去进化能力的生物种群，如《三体》世界观中“安顿”到澳大利亚保留区里“绝育”的人类，被淘汰只是时间问题。]

如果我们能在*写代码*阶段，就把程序准备得*易于后续修改*，那就可以在后续 90% 的*改代码*阶段省下无数时间。

如何让代码易于修改？前人总结出一系列常用的写法，这类写法有助于让后续修改更容易，各自适用于不同的场合，这就是设计模式。

提升可维护性最基础的一点，就是避免重复！

当你有很多地方出现重复的代码时，一旦需要涉及修改这部分逻辑时，就需要到每一个出现了这个逻辑的代码中，去逐一修改。

#fun[例如你的名字，在出生证，身份证，学生证，毕业证，房产证，驾驶证，各种地方都出现了。那么你要改名的话，所有这些证件都需要重新印刷！如果能把他们合并成一个“统一证”，那么只需要修改“统一证”上的名字就行了。]

不过，现实中并没有频繁改名字的需求，这说明：

- 对于不常修改的东西，可以容忍一定的重复。
- 越是未来有可能修改的，就越需要设计模式降重！

例如数学常数 PI = 3.1415926535897，这辈子都不可能出现修改的需求，那写死也没关系。如果要把 PI 定义成宏，只是出于“记不住”“写起来太长了”“复制粘贴麻烦”。所以对于 PI 这种不会修改的东西，降重只是增加*可读性*，而不是*可修改性*。

#tip[但是，不要想当然！需求的千变万化总是超出你的想象。]

例如你做了一个“愤怒的小鸟”游戏，需要用到重力加速度 g = 9.8，你想当然认为 g 以后不可能修改。老板也信誓旦旦向你保证：“没事，重力加速度不会改变。”你就写死在代码里了。

没想到，“愤怒的小鸟”老板突然要求你加入“月球章”关卡，在这些关卡中，重力加速度是 g = 1.6。

如果你一开始就已经把 g 提取出来，定义为常量：

```cpp
struct Level {
    const double g = 9.8;

    void physics_sim() {
        bird.v = g * t; // 假装这里是物理仿真程序
        pig.v = g * t;  // 假装这里是物理仿真程序
    }
};
```

那么要支持月球关卡，只需修改一处就可以了。

```cpp
struct Level {
    double g;

    Level(Chapter chapter) {
        if (chapter == ChapterMoon) {
            g = 1.6;
        } else {
            g = 9.8;
        }
    }

    void physics_sim() {
        bird.v = g * t; // 无需任何修改，自动适应了新的非常数 g
        pig.v = g * t;  // 无需任何修改，自动适应了新的非常数 g
    }
};
```

#fun[小彭老师之前做 zeno 时，询问要不要把渲染管线节点化，方便用户动态编程？张猩猩就是信誓旦旦道：“渲染是一个高度成熟领域，不会有多少修改需求的。”小彭老师遂写死了渲染管线，专为性能极度优化，几个月后，张猩猩羞答答找到小彭老师：“小彭老师，那个，渲染，能不能改成节点啊……”。这个故事告诉我们，甲方的信誓旦旦放的一个屁都不能信。]

=== 用函数封装

函数就是来帮你解决代码重复问题的！要领：

*把共同的部分提取出来，把不同的部分作为参数传入。*

```cpp
void sum(std::vector<int> const &v) {
    int s = 0;
    for (int i = 0; i < v.size(); i++) {
        s += v[i];
    }
    fmt::println("sum of v = {}", s);
}

int main() {
    std::vector<int> a = {1, 2, 3, 4};
    sum(a);
    std::vector<int> b = {5, 6, 7, 8};
    sum(b);
    return 0;
}
```

这样 main 函数里就可以只关心要求和的数组，而不用关心求和具体是如何实现的了。事后我们可以随时把 sum 的内容偷偷换掉，换成并行的算法，main 也不用知道。这就是*封装*，可以把重复的公共部分抽取出来，方便以后修改代码。

#fun[sum 函数相当于，当需要吹空调时，插上空调插座。当需要给手机充电时，插上手机充电器。你不需要关心插座里的电哪里来，“国家电网”会替你想办法解决，想办法优化，想办法升级到绿色能源。你只需要吹着空调给你正在开发的手机 App 优化就行了，大大减轻程序员心智负担。]

=== 要封装，但不要耦合

但是！这段代码仍然有个问题，我们把 sum 求和的结果，直接在 sum 里打印了出来。sum 里写死了，求完和之后只能直接打印，调用者 main 根本无法控制。

这是一种错误的封装，或者说，封装过头了。

#fun[你把手机充电器 (fmt::println) 焊死在了插座 (sum) 上，现在这个插座只能给手机充电 (用于直接打印) 了，不能给笔记本电脑充电 (求和结果不直接用于打印) 了！尽管通过更换充电线 (参数 v)，还可以支持支持安卓 (a) 和苹果 (b) 两种手机的充电，但这样焊死的插座已经和笔记本电脑无缘了。]

=== 每个函数应该职责单一，别一心多用

很明显，“打印”和“求和”是两个独立的操作，不应该焊死在一块。

sum 函数的本职工作是“数组求和”，不应该附赠打印功能。

sum 计算出求和结果后，直接 return 即可。

#fun[如何处理这个结果，是调用者 main 的事，正如“国家电网”不会管你用他提供的电来吹空调还是玩游戏一样，只要不妨碍到其他居民的正常用电。]

```cpp
int sum(std::vector<int> const &v) {
    int s = 0;
    for (int i = 0; i < v.size(); i++) {
        s += v[i];
    }
    return s;
}

int main() {
    std::vector<int> a = {1, 2, 3, 4};
    fmt::println("sum of a = {}", sum(a));
    std::vector<int> b = {5, 6, 7, 8};
    fmt::println("sum of b = {}", sum(b));
    return 0;
}
```

这就是设计模式所说的*职责单一原则*。

=== 二次封装

假设我们要计算一个数组的平均值，可以再定义个函数 average，他可以基于 sum 实现：

```cpp
int sum(std::vector<int> const &v) {
    int s = 0;
    for (int i = 0; i < v.size(); i++) {
        s += v[i];
    }
    return s;
}

double average(std::vector<int> const &v) {
    return (double)sum(v) / v.size();
}

int main() {
    std::vector<int> a = {1, 2, 3, 4};
    fmt::println("average of a = {}", average(a));
    std::vector<int> b = {5, 6, 7, 8};
    fmt::println("average of b = {}", average(b));
    return 0;
}
```

进一步封装一个打印数组所有统计学信息的函数：

```cpp
void print_statistics(std::vector<int> const &v) {
    if (v.empty()) {
        fmt::println("this is empty...");
    } else {
        fmt::println("sum: {}", sum(v));
        fmt::println("average: {}", average(v));
        fmt::println("min: {}", min(v));
        fmt::println("max: {}", max(v));
    }
}

int main() {
    std::vector<int> a = {1, 2, 3, 4};
    print_statistics(a);
    std::vector<int> b = {5, 6, 7, 8};
    print_statistics(b);
    return 0;
}
```

暴露 API 时，要同时提供底层的 API 和高层封装的 API。用户如果想要控制更多细节可以调用底层 API，想要省事的用户可以调用高层封装好的 API。

#tip[高层封装 API 应当可以完全通过调用底层 API 实现，提供高层 API 只是方便初级用户使用和理解。]

#story[
    例如 `libcurl` 就提供了 `curl_easy` 和 `curl_multi` 两套 API。

    - `curl_multi` 提供了超详细的参数，把每个操作分拆成多步，方便用户插手细节，满足高级用户的定制化需求，但太过复杂，难以学习。
    - `curl_easy` 是对 `curl_multi` 的再封装，提供了更简单的 API，但是对具体细节就难以操控了，适合初学者上手。
]

=== Linus 的最佳实践：每个函数不要超过 3 层嵌套，函数体不要超过 24 行

Linux 内核为什么坚持使用 `TAB=8` 为代码风格？

TODO：还在写

== 为什么需要函数式？

你产生了两个需求，分别封装了两个函数：

- `sum` 求所有元素的和
- `product` 求所有元素的积

```cpp
int sum(std::vector<int> const &v) {
    int ret = v[0];
    for (int i = 1; i < v.size(); i++) {
        ret += v[i];
    }
    return ret;
}

int product(std::vector<int> const &v) {
    int ret = v[0];
    for (int i = 1; i < v.size(); i++) {
        ret *= v[i];
    }
    return ret;
}

int main() {
    std::vector<int> a = {1, 2, 3, 4};
    fmt::println("sum: {}", sum(a));
    fmt::println("product: {}", product(a));
    return 0;
}
```

注意到 `sum` 和 `product` 的内容几乎如出一辙，唯一的区别在于：

- `sum` 的循环体为 `+=`；
- `product` 的循环体为 `*=`。

这种函数体内有部分代码重复，但又有特定部分不同，难以抽离。

该怎么复用这重复的部分代码呢？

我们要把 `sum` 和 `product` 合并成一个函数 `generic_sum`。然后通过函数参数，把差异部分（0、`+=`）“注入”到两个函数原本不同地方。

=== 枚举的糟糕用法

如何表示我这个函数是要做求和 `+=` 还是求积 `*=`？

让我们定义枚举：

```cpp
enum Mode {
    ADD, // 求和操作
    MUL, // 求积操作
};

int generic_sum(std::vector<int> const &v, Mode mode) {
    int ret = v[0];
    for (int i = 1; i < v.size(); i++) {
        if (mode == ADD) { // 函数内判断枚举，决定要做什么操作
            ret += v[i];
        } else if (mode == MUL) {
            ret *= v[i];
        }
    }
    return ret;
}

int main() {
    std::vector<int> a = {1, 2, 3, 4};
    fmt::println("sum: {}", generic_sum(a, ADD)); // 用户指定他想要的操作
    fmt::println("product: {}", generic_sum(a, MUL));
    return 0;
}
```

然而，如果用户现在想要求数组的*最大值*呢？

枚举中还没有实现最大值的操作……要支持，就得手忙脚乱地去修改 `generic_sum` 函数和 `Mode` 枚举原本的定义，真麻烦！

```cpp
enum Mode {
    ADD,
    MUL,
    MAX, // **改**
};

int generic_sum(std::vector<int> const &v, Mode mode) {
    int ret = v[0];
    for (int i = 1; i < v.size(); i++) {
        if (mode == ADD) {
            ret += v[i];
        } else if (mode == MUL) {
            ret *= v[i];
        } else if (mode == MAX) { // **改**
            ret = std::max(ret, v[i]); // **改**
        }
    }
    return ret;
}

int main() {
    std::vector<int> a = {1, 2, 3, 4};
    generic_sum(a, MAX); // **改**
    return 0;
}
```

#tip[我用 `// **改**` 指示了所有需要改动的地方。]

为了增加一个求最大值的操作，就需要三处分散在各地的改动！

不仅如此，还容易抄漏，抄错，比如 `MAX` 不小心打错成 `MUL` 了，自己却没发现，留下 BUG 隐患。

这样写代码的方式，心智负担极大，整天就提心吊胆着东一块，西一块的散装代码，担心着有没有哪个地方写错写漏，严重妨碍了开发效率。

并且写出来的代码也不能适应需求的变化：假如我需要支持 `MIN` 呢？又得改三个地方！这违背了设计模式的*开闭原则*。

/ 开闭原则: 对扩展开放，对修改封闭。指的是软件在适应需求变化时，应尽量通过*扩展代码*来实现变化，而不是通过*修改已有代码*来实现变化。

使用枚举和 if-else 实现多态，难以扩展，还要一直去修改原函数的底层实现，就违背了*开闭原则*。

=== 函数式编程光荣救场

如果我们可以“注入”代码就好了！能否把一段“代码”作为 `generic_sum` 函数的参数呢？

代码，实际上就是函数，注入代码就是注入函数。我们先定义出三个不同操作对应的函数：

```cpp
int add(int a, int b) {
    return a + b;
}

int mul(int a, int b) {
    return a * b;
}

int max(int a, int b) {
    return std::max(a, b);
}
```

然后，把这三个小函数，作为另一个大函数 `generic_sum` 的参数就行！

```cpp
int generic_sum(std::vector<int> const &v, auto op) {
    int ret = v[0];
    for (int i = 1; i < v.size(); i++) {
        // 函数作者无需了解用户指定的“操作”具体是什么
        // 只需要调用这一“操作”，得到结果就行
        ret = op(ret, v[i]);
    }
    return ret;
}

int main() {
    std::vector<int> a = {1, 2, 3, 4};
    // 用户无需关心函数的具体实现是什么
    // 只需随心所欲指定他的“操作”作为参数
    generic_sum(a, add);
    generic_sum(a, product);
    generic_sum(a, max);
    return 0;
}
```

责任明确了，我们成功把一部分细节从 `generic_sum` 中进一步抽离。

- 库作者 `generic_sum` 不必了解 `main` 的操作具体是什么，他只负责利用这个操作求“和”。
- 库用户 `main` 不必了解 `generic_sum` 如何实现操作累加，他只管注入“如何操作”的代码，以函数的形式。

=== 我用了 C++20 的函数参数 auto 语法糖

```cpp
int generic_sum(std::vector<int> const &v, auto op) P
}
```

这里的参数 op 类型声明为 auto，效果就是，op 这个参数现在能接受任意类型的对象了（包括函数！）

```cpp
int generic_sum(std::vector<int> const &v, auto op) {
    ...
}
```

#detail[准确的说，`auto op` 参数的效果是使 `generic_sum` 变为一个*模板函数*，其中 op 参数变成了模板参数，能够接受任意类型了。而写明类型的参数 `std::vector<int> const &v` 就没有任何额外效果，就只能接受 `vector<int>` 而已。]

如果你不支持 C++20 的话，需要显式写出 `template`，才能实现同样的效果：

```cpp
template <typename Op>
int generic_sum(std::vector<int> const &v, Op op) {
    ...
}
```

#fun[C++11：auto 只能用于定义变量；C++14：函数返回类型可以是 auto；C++17：模板参数也可以 auto；C++20：函数参数也可以是 auto 了；（狂想）C++47：auto 现在是 C++47 的唯一关键字，用户只需不断输入 auto-auto-auto，编译器内建人工智能自动识别你的意图生成机器码。]

=== 函数也是对象！

在过去的*面向对象编程范式*中，函数（代码）和对象（数据）被*割裂*开来，他们愚昧地认为*函数不是对象*。

*函数式编程范式*则认为：*函数也是一种变量，函数可以作为另一个函数的参数！*

#fun[Function lives matter!]

#detail[面向对象就好比计算机的“哈佛架构”，代码和数据割裂，代码只能单方面操作数据。函数式就好比“冯诺依曼架构”，代码也是数据。看似会导致低效，实则大大方便了动态加载新程序，因而现在的计算机基本都采用了“冯诺依曼架构”。]

总之，函数也是对象，被亲切地尊称为*函数对象*。

=== C++11 引入 Lambda 语法糖

C++98 时代，人们还需要单独跑到 `main` 外面，专门定义 `add`、`mul`、`max` 函数。弄得整个代码乱哄哄的，非常麻烦。

```cpp
int add(int a, int b) {
    return a + b;
}

int mul(int a, int b) {
    return a * b;
}

int max(int a, int b) {
    return std::max(a, b);
}

int main() {
    std::vector<int> a = {1, 2, 3, 4};
    generic_sum(a, add);
    generic_sum(a, product);
    generic_sum(a, max);
    return 0;
}
```

C++11 引入了 *Lambda 表达式*语法，允许你就地创建一个函数。

```cpp
int main() {
    std::vector<int> a = {1, 2, 3, 4};

    auto add = [](int a, int b) {
        return a + b;
    };
    auto mul = [](int a, int b) {
        return a * b;
    };
    auto max = [](int a, int b) {
        return std::max(a, b);
    };

    generic_sum(a, add);
    generic_sum(a, product);
    generic_sum(a, max);
    return 0;
}
```

不用往 `main` 外面塞垃圾了，一清爽。

更进一步，我们甚至不用定义变量，直接把 Lambda 表达式写在 `generic_sum` 的参数里就行了！

```cpp
int main() {
    std::vector<int> a = {1, 2, 3, 4};

    generic_sum(a, [](int a, int b) {
        return a + b;
    });
    generic_sum(a, [](int a, int b) {
        return a * b;
    });
    generic_sum(a, [](int a, int b) {
        return std::max(a, b);
    }); // **改**
    return 0;
}
```

#tip[以上写法都是等价的。]

要支持一个新操作，只需修改一处地方：在调用 `generic_sum` 时就地创建一个函数。随叫随到，不用纠结于“起名强迫症”，是不是很方便呢？

#detail[准确的说，Lambda 创建的是函数对象 (function object) 或称仿函数 (functor) 而不是传统意义上的函数。]

#story[其实 C++98 时代人们就已经大量在用 `operator()()` 模拟函数对象了，著名的第三方库 Boost 也封装了各种函数式常用的容器和工具。C++11 才终于把*函数对象*这个概念转正，并引入了更方便的 Lambda 语法糖。]

#fun[即使是面向对象的头号孝子 Java，也已经开始引入函数式的 Lambda 语法糖，C\# 的 LINQ 更是明目张胆的致敬 map-reduce 全家桶，甚至 C 语言用户也开始玩各种函数指针回调……没办法，函数式确实方便呀！]

=== 依赖注入原则

函数对象 `op` 作为参数传入，让 `generic_sum` 内部去调用，就像往 `generic_sum` 体内“注入”了一段自定义代码一样。

这可以让 `generic_sum` 在不修改本体的情况下，通过修改“注入”部分，轻松扩展，满足*开闭原则*。

更准确的说，这体现的是设计模式所要求的*依赖注入原则*。

/ 依赖注入原则: 一个封装好的函数或类，应该尽量依赖于抽象接口，而不是依赖于具体实现。这可以提高程序的灵活性和可扩展性。

四大编程范式都各自发展出了*依赖注入原则*的解决方案：

- 面向过程编程范式中，*函数指针*就是那个抽象接口。
- 面向对象编程范式中，*虚函数*就是那个抽象接口。
- 函数式编程范式中，*函数对象*就是那个抽象接口。
- 模板元编程范式中，*模板参数*就是那个抽象接口。

同样是把抽象接口作为参数，同样解决可扩展问题。

函数指针贴近底层硬件，虚函数方便整合多个接口，函数对象轻量级、随地取用，模板元有助高性能优化，不同的编程范式殊途同归。

=== 低耦合，高内聚

依赖注入原则可以减少代码之间的耦合度，大大提高代码的灵活性和可扩展性。

/ 耦合度: 指的是一个模块、类、函数和其他模块、类、函数之间的关联程度。耦合度越低，越容易进行单元测试、重构、复用和扩展。

#fun[高耦合度的典型是“牵一发而动全身”。低耦合的典范是蚯蚓，因为蚯蚓可以在任意断面切开，还能活下来，看来蚯蚓的身体设计非常“模块化”呢。]

通常来说，软件应当追求低耦合度，适度解耦的软件能更快适应需求变化。但过度的低耦合也会导致代码过于分散，不易阅读和修改，甚至可能起到反效果。

#tip[若你解耦后，每次需求变化要改动的地方变少了，那就是合理的解耦。若你过分解耦，代码东一块西一块，以至于需求变化时需要到处改，比不解耦时浪费的时间还要多，那就是解耦过度。]

#fun[完全零耦合的程序每个函数互不联系，就像把蚯蚓拆散成一个个独立的细胞一样。连初始需求“活着”都实现不了，谈何适应需求变化？所以解耦也切勿矫枉过正。]

为了避免解耦矫枉过正，人们又提出了内聚的概念，并规定解耦的前提是：不耽误内聚。耽误到内聚的解耦，就只会起到降低可维护性的反效果了。

/ 内聚: 指的是同一个模块、类、函数内部各个元素之间的关联程度。内聚度越高，功能越独立，越方便集中维护。

#fun[例如，人的心脏专门负责泵血，肝脏只负责解毒，这就是高内聚的人体器官。若人的心脏还要兼职解毒，肝脏还兼职泵血，看似好像是增加了“万一心脏坏掉”的冗余性，实际上把“泵血”这一功能拆散到各地，无法“集中力量泵大血”了。]

#detail[人类的大脑和 CPU 一样，也有“缓存局域性 (cache-locality)”的限制：不能同时在很多个主题之间快速切换，无论是时间上的还是空间上的割裂 (cache-miss)，都会干扰程序员思维的连贯性，从而增大心智负担。]

好的软件要保持低耦合，同时高内聚。

#fun[就像“民主集中制”一样，既要监督防止大权独揽，又要集中力量办一个人办不成的大事。]

=== 与传统面向对象的对比

传统的面向对象同样可以用*虚函数接口类*模拟*函数对象*一样的功能，只不过没有 lambda 和闭包的语法加持，写起来非常繁琐，就和在 C 语言里“模拟”面向对象一样。

#fun[为了这么小的一个代码块，单独定义一个类，就像妈妈开一架“空中战车” A380 只是为了接你放学一样，等你值好机的时间我自己走都走到了。而函数式中，用 lambda 就地定义函数对象，相当于随地抓来一台共享单车开走。]

```cpp
struct OpBase { // 面向对象：遇事不决先定义接口……
    virtual int operate(int a, int b) = 0;
    virtual ~OpBase() = default;
};

struct OpAdd : OpBase {
    int compute(int a, int b) override {
        return a + b;
    }
};

struct OpMul : OpBase {
    int compute(int a, int b) override {
        return a * b;
    }
};

struct OpMax : OpBase {
    int compute(int a, int b) override {
        return std::max(a, b);
    }
};

int generic_sum(std::vector<int> const &v, OpBase *op) {
    int ret = v[0];
    for (int i = 1; i < v.size(); ++i) {
        ret = op.compute(ret, v[i]); // 写起来也麻烦，需要调用他的成员函数，成员函数又要起名……
    }
    delete op;
    return ret;
}

int main() {
    std::vector<int> a = {1, 2, 3, 4};

    generic_sum(a, new OpAdd());
    generic_sum(a, new OpMul());
    generic_sum(a, new OpMax());
    return 0;
}
```

不仅需要定义一堆类，接口类，实现类，继承来继承去，还需要管理讨厌的指针，代码量翻倍，没什么可读性，又影响运行效率。

#fun[3 年 2 班小彭同学，你的妈妈开着 A380 来接你了。]

而现代 C++ 只需 Lambda 语法就地定义函数对象，爽。

```cpp
    generic_sum(a, [](int a, int b) {
        return a + b;
    });
    generic_sum(a, [](int a, int b) {
        return a * b;
    });
    generic_sum(a, [](int a, int b) {
        return std::max(a, b);
    });
```

=== 函数对象在模板加持下静态分发

刚刚，我们的实现用了 `auto op` 做参数，这等价于让 `generic_sum` 变成一个模板函数。

```cpp
int generic_sum(std::vector<int> const &v, auto op);

// 不支持 C++20 时的替代写法：
template <typename Op>
int generic_sum(std::vector<int> const &v, Op op);
```

这意味着每当用户指定一个新的函数对象（lambda）时，`generic_sum` 都会重新实例化一遍。

```cpp
    generic_sum(a, [](int a, int b) {
        return a + b;
    });
    generic_sum(a, [](int a, int b) {
        return a * b;
    });
    generic_sum(a, [](int a, int b) {
        return std::max(a, b);
    });
```

编译后，会变成类似于这样：

```cpp
    generic_sum<add>(a);
    generic_sum<mul>(a);
    generic_sum<max>(a);
```

会生成三份函数，每个都是独立编译的：

```cpp
int generic_sum<add>(std::vector<int> const &v) {
    int ret = v[0];
    for (int i = 1; i < v.size(); ++i) {
        ret = add(ret, v[i]);
    }
    return ret;
}
int generic_sum<mul>(std::vector<int> const &v) {
    int ret = v[0];
    for (int i = 1; i < v.size(); ++i) {
        ret = mul(ret, v[i]);
    }
    return ret;
}
int generic_sum<max>(std::vector<int> const &v) {
    int ret = v[0];
    for (int i = 1; i < v.size(); ++i) {
        ret = max(ret, v[i]);
    }
    return ret;
}
```

这允许编译器为每个版本的 `generic_sum` 单独做优化，量身定制最优的代码。

例如 `add` 这个函数对象，因为只在 `generic_sum<add>` 中使用了，会被被编译器自动内联，不会产生函数调用和跳转的指令，各自优化成单独一条加法 / 乘法 / 最大值指令等。

#detail[比如，编译器会检测到 `+=` 可以矢量化，于是用 `_mm_add_epi32` 替代了。同理，mul 则用 `_mm_mullo_epi32` 替代，max 则用 `_mm_max_epi32` 替代等，各自分别生成了各自版本最优的代码。而如果是普通的函数指针，不会生成三份量身定做的实例，无法矢量化（有一种例外，就是编译器检测到了 `generic_sum` 似乎只有这三种可能参数，然后做了 IPO 优化，但并不如模板实例化一样稳定强制）。]

为三种不同的 op 参数分别定做三份。虽然增加了编译时间，膨胀了生成的二进制体积；但生成的机器码是分别针对每种特例一对一深度优化的，更高效。

#story[例如矩阵乘法（gemm）的最优算法，对于不同的矩阵大小和形状是不同的。著名的线性代数库 CUBLAS 和 MKL 中，会自动根据用户输入的矩阵形状，选取最优的算法。也就是说，CUBLAS 库里其实存着适合各种矩阵大小排列组合的算法代码（以 fatbin 格式存储在二进制中）。当调用矩阵乘法时，自动查到最适合的一版来调用给你。类似 gemm，还有 gemv、spmv……所有的矩阵运算 API 都经历了这样的“编译期”暴力排列组合，只为“运行时”释放最大性能！这也导致编译好的 cublas.dll 文件来到了恐怖的 20 MB 左右，而我们称之为高效。]

=== 函数对象也可在 function 容器中动态分发

`std::function<int(int, int)>`

```cpp
int generic_sum(std::vector<int> const &v,
                std::function<int(int, int)> op) {
    int ret = v[0];
    for (int i = 1; i < v.size(); ++i) {
        ret = op(ret, v[i]); // 写起来和模板传参时一样无感
    }
    // 无需指针，无需 delete，function 能自动管理函数对象生命周期
    return ret;
}
```

函数式编程，能在静态与动态之间轻松切换，高性能与灵活性任君选择。

可以在需要性能的*瓶颈代码*中用模板传参，编译期静态分发，多次量身定做，提高运行时性能。

/ 瓶颈代码: 往往一个程序 80% 的时间花在 20% 的代码上。这 20% 是在程序中频繁执行的、计算量大的、或者调用特别耗时的函数。针对这部分瓶颈代码优化即可，而剩余的 80% 打酱油代码，大可以怎么方便怎么写。

可以在性能无关紧要的顶层业务逻辑中用 function 容器传参，运行时动态分发，节省编译体积，方便持久存储，灵活易用。

#tip[例如上面的函数，如果我们突然想要高性能了，只需把 `std::function<int(int, int)> op` 轻轻改为 `auto op` 就轻松切换到静态分发模式了。]

而虚函数一旦用了，基本就只能动态分发了，即使能被 IPO 优化掉，虚表指针也永远占据着一个 8 字节的空间，且永远只能以指针形式传来传去。

#detail[一种静态分发版的虚函数替代品是 CRTP，他基于模板元编程，但与虚函数之间切换困难，不像函数对象那么无感，之后的模板元专题章节中会专门介绍。]

=== 函数对象的动态分发用于多线程任务队列

=== 函数对象的重要机制：闭包

=== 函数指针是 C 语言陋习，改掉

== bind 为函数对象绑定参数

```cpp
int hello(int x, int y) {
    fmt::println("hello({}, {})", x, y);
    return x + y;
}

int main() {
    fmt::println("main 调用 hello(2, 3) 结果：{}", hello(2, 3));
    fmt::println("main 调用 hello(2, 4) 结果：{}", hello(2, 4));
    fmt::println("main 调用 hello(2, 5) 结果：{}", hello(2, 5));
    return 0;
}
```

```cpp
int hello(int x, int y) {
    fmt::println("hello({}, {})", x, y);
    return x + y;
}

int main() {
    fmt::println("main 调用 hello2(3) 结果：{}", hello2(3));
    fmt::println("main 调用 hello2(4) 结果：{}", hello2(4));
    fmt::println("main 调用 hello2(5) 结果：{}", hello2(5));
    return 0;
}
```
