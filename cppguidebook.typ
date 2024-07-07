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

#tip[例如 C++20 早已允许 lambda 表达式捕获 structural-binding 变量，而 Clang 至今还没有支持，尽管 Clang 已经支持了很多其他 C++20 特性。]

- Apple Clang 是苹果公司自己魔改的 Clang 版本，只在 MacOS 系统上可用，支持 Objective-C 和 Swift 语言。但是版本较官方 Clang 落后一些，很多新特性都没有跟进，基本上只有专门伺候苹果的开发者会用。

#tip[GCC 和 Clang 也支持 Objective-C。]

- MSVC 是 Windows 限定的编译器，提供了很多 MSVC 特有的扩展。也有人在 Clang 上魔改出了 MSVC 兼容模式，兼顾 Clang 特性的同时，支持了 MSVC 的一些特性（例如 `__declspec`），可以编译用了 MSVC 特性的代码，即 `clang-cl`，在最新的 VS2022 IDE 中也集成了 `clang-cl`。值得注意的是，MSVC 的优化能力是比较差的，比 GCC 和 Clang 都差，例如 MSVC 几乎总是假定所有指针 aliasing，这意味着当遇到很多指针操作的循环时，几乎没法做循环矢量化。但是也使得未定义行为不容易产生 Bug，另一方面，这也导致一些只用 MSVC 的人不知道某些写法是未定义行为。

- Intel C++ compiler 是英特尔开发的 C++ 编译器，由于是硬件厂商开发的，特别擅长做性能优化。但由于更新较慢，基本没有更上新特性，也没什么人在用了。

#tip[最近他们又出了个 Intel DPC++ compiler，支持最新的并行编程领域特定语言 SyCL。]

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

#fun[牛头人笑话：“如果你不知道一个人是用的什么标准库，那么你可以猜他用的是 libstdc++。即使他的编译器是 Clang，他用的大概率依然是 libstdc++。”]

=== 标准库的调试模式

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

#tip[严格来说，是 C++ 运行时调用了 `main` 函数，但目前先理解为#quote[操作系统调用了 `main` 函数]也无妨。]

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

#tip[对于没有返回值（返回类型为 `void`）的函数，可以省略 `return`。]

#warn[对于有返回值的函数，必须写 return 语句，否则程序出错。]

= 函数式编程

== 为什么需要函数？

```cpp
int main() {
    std::vector<int> a = {1, 2, 3, 4};
    int sum = 0;
    for (int i = 0; i < a.size(); i++) {
        sum += a[i];
    }
    fmt::println("sum = {}", sum);
    return 0;
}
```

这是一个计算数组求和的简单程序。

但是，他只能计算数组 a 的求和，无法复用。

如果我们有另一个数组 b 也需要求和的话，就得把整个求和的 for 循环重新写一遍：

```cpp
int main() {
    std::vector<int> a = {1, 2, 3, 4};
    int sum = 0;
    for (int i = 0; i < a.size(); i++) {
        sum += a[i];
    }
    fmt::println("sum of a = {}", sum);
    std::vector<int> b = {5, 6, 7, 8};
    sum = 0;
    for (int i = 0; i < a.size(); i++) {
        sum += b[i];
    }
    fmt::println("sum of b = {}", sum);
    return 0;
}
```

这就出现了程序设计的大忌：代码重复。

#fun[例如，你有吹空调的需求，和充手机的需求。你为了满足这两个需求，购买了两台发电机，分别为空调和手机供电。第二天，你又产生了玩电脑需求，于是你又购买一台发电机，专为电脑供电……]

重复的代码不仅影响代码的*可读性*，也增加了*维护*代码的成本。

+ 看起来乱糟糟的，信息密度低，让人一眼看不出代码在干什么的功能
+ 很容易写错，看走眼，难调试
+ 复制粘贴过程中，容易漏改，比如这里的 `sum += b[i]` 可能写成 `sum += a[i]` 而自己不发现
+ 改起来不方便，当我们的需求变更时，需要多处修改，比如当我需要改为计算乘积时，需要把两个地方都改成 `sum *=`
+ 改了以后可能漏改一部分，留下 Bug 隐患
+ 敏捷开发需要反复修改代码，比如你正在调试 `+=` 和 `-=` 的区别，看结果变化，如果一次切换需要改多处，就影响了调试速度

=== 狂想：没有函数的世界？

如果你还是喜欢“一本道”写法的话，不妨想想看，完全不用任何标准库和第三方库的函数和类，把 `fmt::println` 和 `std::vector` 这些函数全部拆解成一个个系统调用。那这整个程序会有多难写？

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
    int sum = 0;
    for (int i = 0; i < 4; i++) {
        sum += a[i];
    }
    char buffer[64];
    buffer[0] = 's';
    buffer[1] = 'u';
    buffer[2] = 'm';
    buffer[3] = ' ';
    buffer[4] = '=';
    buffer[5] = ' '; // 例如，如果要修改此处的提示文本，甚至需要修改后面的 len 变量...
    int len = 6;
    int x = sum;
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
    int sum = 0;
    for (int i = 0; i < 4; i++) {
        sum += b[i];
    }
    len = 6;
    x = sum;
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

#fun[软件就像生物，要不断进化，软件不更新不维护了等于死。而若你的代码逐渐变得臃肿难以修改，无法适应新需求，那你的代码就像已经失去进化能力的生物种群，如《三体》世界观中“安置”到澳大利亚保留区里“绝育”的人类，被淘汰只是时间问题。]

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

那么支持月球关卡，只需要修改一处就可以了。

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
    int sum = 0;
    for (int i = 0; i < v.size(); i++) {
        sum += v[i];
    }
    fmt::println("sum of v = {}", sum);
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

如何处理这个结果，是调用者 main 的事，正如“国家电网”不会管你用他提供的电来吹空调还是玩游戏一样，只要不妨碍到其他居民的正常用电。

```cpp
int sum(std::vector<int> const &v) {
    int sum = 0;
    for (int i = 0; i < v.size(); i++) {
        sum += v[i];
    }
    return sum;
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
    int sum = 0;
    for (int i = 0; i < v.size(); i++) {
        sum += v[i];
    }
    return sum;
}

int average(std::vector<int> const &v) {
    return sum(v) / v.size();
}

int main() {
    std::vector<int> a = {1, 2, 3, 4};
    fmt::println("average of a = {}", average(a));
    std::vector<int> b = {5, 6, 7, 8};
    fmt::println("average of b = {}", average(b));
    return 0;
}
```

== 为什么需要函数式？

== 函数对象

== bind

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
