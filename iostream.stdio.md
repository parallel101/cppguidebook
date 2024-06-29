[toc]

# 标准输入输出流 ^[iostream.stdio]

流是 C++ 中处理输入输出的机制。

C++ 标准库提供了四个流对象，分别是：

- `std::cin`：输入流
- `std::cout`：输出流
- `std::cerr`：标准错误输出流（无缓冲）
- `std::clog`：标准错误输出流

它们都定义在标准库专属命名空间 `std` 中。

## 头文件

使用前，导入头文件 ^[pp.header]：

```cpp
#include <iostream>
```

## 全局变量

该头文件将引入下列全局变量 ^[lang.global]：

```cpp
std::istream std::cin;
std::ostream std::cout;
std::ostream std::cerr;
std::ostream std::clog;
std::wistream std::cin;
std::wostream std::cout;
std::wostream std::cerr;
std::wostream std::clog;
```

宽字符流的讲解将放到一个专题中，见 ^[locale.wide.stdio]。

## 介绍

`std::cin` 是输入流 ^[iostream.istream]。

`std::cout`、`std::cerr`、`std::clog` 是输出流 ^[iostream.ostream]。

一般来说，这些流操作的对象是终端 ^[os.terminal]，用于在终端上输入和输出文本 ^[string.text]。

> 终端 (terminal) 就是那个 “C 语言程序启动后弹出的黑色窗口”，有时也被称为控制台 (console)。

输入流的用法是 `std::cin >> 变量`。

用户在终端上键盘输入文本或数字等内容后，程序使用输入流对象 `std::cin` 读取，就可以把用户的输入读取到变量中。

> 关于流内部如何将用户输入的文本解析为数字变量，参见 ^[iostream.format.in]。

输出流的用法是 `std::cout << 变量` 或 `std::cout << "字符串"`。

程序使用输出流后，变量或字符串的内容将会追加到终端光标处。通常用于向用户反馈程序运行进度，计算结果等。

此处的 `<<` 是 C++ 的运算符重载 ^[lang.class.operator] 语法。

标准库为 `std::istream` 类重载了 `>>` 运算符，目的是为了看起来更“高级感”，就好像是移位运算符在“搬运”出东西来一样。

而 `std::cin` 就是一个 `std::istream` 类，因此 `std::cin` 对象支持 `>>` 运算符。

同样地，标准库为 `std::ostream` 类重载了 `<<` 运算符，而 `std::cout` 都是 `std::ostream` 类，因此他们都支持 `>>` 运算符。

这是一个糟糕的设计 ^[design.bad]，带来了巨大的混淆。看起来是“移位”的运算符，实际上做的事却与“移位”毫无关系：输入/输出变量。

```cpp
int i;
std::cin >> i;
std::cout << i;
```

等价于调用他们的成员函数 `operator>>` 和 `operator<<`：

```cpp
int i;
std::cin.operator>>(i);
std::cout.operator<<(i);
```

此处 `std::cin` 的 `operator>>` 接受的参数是引用 ^[lang.ref]，为的是让 `operator>>` 函数内部可以修改 i 的值，这种传参方式称为按引用传参 (pass-by-reference) ^[lang.func.call.arg]，此处按引用传参是为了让 cin 可以返回读到的整数值写入 i 变量。

如果让我来设计 `<iostream>` 的话，我会这样设计接口 ^[design.api]：

```cpp
// 理想中的 iostream，现实中并不存在这样的写法
int i;
std::cin.get(i);
std::cout.put(i);
```

## 输出流用法

接下来的输出流用法讲解将以 `std::cout` 为例，对于 `std::cerr` 和 `std::clog` 同样适用。

### 打印字符串

代码：

```cpp
#include <iostream>

int main() {
    std::cout << "你好，世界";
    return 0;
}
```

程序运行时会在终端上显示出 `你好，世界`。

此处程序源码中的 `"你好，世界"` 是字符串字面量 ^[string.literial]。

运行结果：

```
你好，世界
```

### 打印变量

代码：

```cpp
#include <iostream>

int main() {
    int i = 42;
    std::cout << i;
    return 0;
}
```

运行结果：

```cpp
42
```

### 换行符

代码：

```cpp
#include <iostream>

int main() {
    std::cout << "你好\n世界";
    return 0;
}
```

程序运行时会在终端上显示 `你好` 然后换一行，再输出 `世界`。此处字符串中的 `\n` 表示换行符，输出该字符后，终端上的文字会另起一行书写。

运行结果：

```
你好
世界
```

### 更多转义符

换行符 `\n` 是字符串字面量所支持的转义符的一种，详见 ^[string.literial.escape]。

如果真的要输出 `\n` 的话，需要写 `"\\n"`，此处 `\\` 也是一种转义符，表示原原本本的 `\` 自己。

```cpp
#include <iostream>

int main() {
    std::cout << "你好\\n世界";
    return 0;
}
```

运行结果：

```
你好\n世界
```

由于两边的 `"` 用于表示字符串字面量的边界，如果真的要输出 `"` 的话，需要写 `"\""`，此处 `\"` 也是一种转义符，表示原原本本的 `"` 自己，而不是字符串的结束。

```cpp
#include <iostream>

int main() {
    std::cout << "这是一个引号：\"，他不会被编译器误解为字符串常量的结束";
    return 0;
}
```

运行结果：

```
这是一个引号："，他不会被编译器误解为字符串常量的结束
```

### 连续换行

建议每条 cout 打印的消息末尾，都加上换行符，否则多条消息之间没有分隔，会黏在一起：

```cpp
#include <iostream>

int main() {
    std::cout << "第一条消息";
    std::cout << "第二条消息";
    return 0;
}
```

运行结果：

```
第一条消息第二条消息
```

为第一条消息和第二条消息末尾加上换行符，就能保证每条消息总是分开：

```cpp
#include <iostream>

int main() {
    std::cout << "第一条消息\n";
    std::cout << "第二条消息\n";
    return 0;
}
```

```
第一条消息
第二条消息
```

### 连续使用 `<<` 运算符

> 小贴士：如果在多线程中并行 `<<` 输出到 cout，可能会出现黏行现象，这是因为每一条单独的 `<<` 操作是原子的（cout 是自带锁的同步流），而连续多条 `<<` 之间并不，即使是同一行上写的连续 `<<` 也不行，原因分析和解决方案见 ^[thread.osyncstream]。

## 输入流

`std::cin` 是标准输入流，只读。对应于 C 语言的 `stdin` ^[cstdio.stdio]。

通常用于读取用户在终端输入的文本。

### 读入变量

```cpp
#include <iostream>

int main() {
    int a;
    int b;
    std::cout << "请输入数字：\n";
    std::cin >> a >> b;
    int res = a + b;
    std::cout << "数字之和为：" << res << '\n';
}
```

```
$ ./myprogram
请输入数字：
$ 1 2
数字之和为：3
```

### 输入流的行缓冲机制

输入流的行缓冲 ^[os.terminal.buf] 指的是，由于操作系统的设计原因：

用户输入的字符，并不会马上被 `std::cin` 收到，而是等用户按下回车键 (enter) 后才会把输入的文本发送给程序。

这样设计的初衷是为了让用户可以在输入一部分文本后，检查输入是否正确，按下回车键 (backspace) 删除输错的部分，然后再输入剩下的文本，确认无误后再按下回车键。

糟糕的是，输入流的行缓冲是操作系统层面实现的，并不是标准库的功劳。

因此，输入流的行缓冲与输出流的行缓冲有着很大不同：

- 输出流的缓冲机制是标准库实现的，可以通过调用 `std::ostream` 的接口关闭缓冲。

- 输入流的缓冲机制是操作系统实现的，需要调用操作系统的底层 API 才能关闭，例如利用 termios ^[os.terminal.termios]。

关于输入流缓冲机制的进一步详细说明，以及关闭输入缓冲的方法，参见 ^[os.terminal.buf]。对于 Windows 系统，可以用无缓冲的 getch 系列函数，参见 ^[win.crt.conio]。

## 输出流

`std::cout` 是标准输出流，只写。对应于 C 语言的 `stdout`。

通常用于将文本打印到终端上，也可以被重定向到指定文件。

`std::clog` 是标准错误输出流，只写。对应于 C 语言的 `stderr`。

通通常用于将文本打印到终端上。常用于输出日志和调试信息。

`std::cerr` 是标准错误输出流，只写。对应于 C 语言的 `stderr`。

> 稍后会说明为什么要区分错误输出与常规输出。

### 输出流的行缓冲机制

`std::cout` 和 `std::clog` 输出流具有行缓冲 ^[iostream.buf]。

TODO

### `std::cerr` 与 `std::clog` 的区别

`std::cerr` 禁用了行缓冲，每次 `<<` 后都会自动强制刷新缓冲区 (flush buffer)。

而 `std::clog` 不会强制刷新缓冲，需要手动 `std::clog << std::flush` 才能刷新。

由于所有流默认都开启了行缓冲，即检测到输出了换行符 `\n` 时会自动 flush 刷新缓冲区，因此只要你每次 clog 后都输出 `\n` 换行，那和 cerr 就没有区别，反而因为 `std::clog` 避免了多余的反复刷新，提升了性能。

这样设计的初衷是方便程序要打印的错误信息及时输出到终端上显示，让用户及时看到。

否则如果程序调用 clog 来输出，由于 clog 会把文本缓冲起来，稍后才一次性提交给操作系统，如果程序在缓冲尚未提交期间异常崩溃而不是正常退出 ^[life.atexit]，clog 缓冲的部分文本就会丢失。

关于输出流的缓冲与刷新机制，以及关闭输出缓冲的方法，参见 ^[iostream.buf]。

### 不推荐 `std::endl`

TODO

## 重定向

### 输入流重定向

在操作系统提供的命令提示符或 Shell ^[os.shell] 中，输入流可以被重定向 ^[os.io.redirect] 到指定文件。

创建一个文件 `file.txt`，其内容为：

```
1 2
```

使用 Shell 的 `<` 语法，可以重定向输入流为 `file.txt`：

```
$ ./myprogram < file.txt
请输入数字：
数字之和为：3
```

这样用户就无需在终端键入任何内容，程序自动从指定的输入文件 `file.txt` 读取了输入。

### 输出流重定向

TODO

> 操作系统实现重定向的机制和原理详见 ^[os.io.redirect]。

> 进一步，还可以利用操作系统的管道机制，把一个程序的输出流作为另一个程序的输入流，详见 ^[os.io.pipe]。

### 分离常规输出与错误输出的意义

```cpp
int main() {
    std::cout << "这是一条 cout 消息\n";
    std::cerr << "这是一条 cerr 消息\n";
}
```

TODO

### 程序内主动重定向

TODO

## 行缓冲升级为全缓冲

```cpp
std::ios::sync_with_stdio(false);
```

TODO

## 不推荐 `using namespace std`

你可能看到有些人的 C++ 代码中使用 `cout` 时不用加 `std::` 前缀。

这是因为他们在导入头文件后加入了一行 `using namespace std` ^[namespace.using]，使 `std::` 命名空间下的所有符号都暴露到了全局命名空间。

```cpp
#include <iostream>

using namespace std; // 导入 std 命名空间

int main() {
    cout << "你好，世界\n";  // 不用写 std::cout，只写 cout 也可以了
    return 0;
}
```

运行结果：

```
你好，世界
```

不推荐 `using namespace std`，因为 `std::` 中的符号又多又杂，可能与你的程序中定义的符号冲突。

现实工程中人们都会写出 `std::` 前缀，这有助于你区分哪些是标准库的，哪些是自己定义的符号。
