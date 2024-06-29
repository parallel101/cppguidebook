[toc]

# 标准输入输出流 [iostream.stdio]

流是 C++ 中处理输入输出的机制。

C++ 标准库提供了四个流对象，分别是：

- `std::cin`：输入流
- `std::cout`：输出流
- `std::cerr`：标准错误输出流（无缓冲）
- `std::clog`：标准错误输出流

它们都定义在命名空间 `std` 中。

## 头文件 ^[[pp.header]]

```cpp
#include <iostream>
```

## 全局变量 ^[[lang.global]]

```cpp
std::istream std::cin;
std::ostream std::cout;
std::ostream std::cerr;
std::ostream std::clog;
```

## 介绍

`std::cin` 是输入流 (`std::istream`)。

`std::cout`、`std::cerr`、`std::clog` 是输出流 (`std::ostream`)。

一般来说，这些流操作的对象是终端 ^[[os.terminal]]，用于在控制台上输入输出文本。

> 终端 (terminal) 就是那个 “C 语言程序启动后弹出的黑色窗口”，有时也被称为控制台 (console)。

输入流的用法是 `std::cin >> 变量`。

用户在终端上键盘输入文本或数字等内容后，程序使用输入流就可以读取到用户的输入。

输出流的用法是 `std::cout << 变量` 或 `std::cout << "字符串"`。

程序使用输出流后，变量或字符串的内容将会追加到终端光标处。通常用于向用户反馈程序运行进度，计算结果等。

此处的 `<<` 是 C++ 的运算符重载 ^[[lang.class.operator]] 语法，标准库为 `std::istream` 类重载了 `<<` 运算符，目的是为了让使用者看起来更便捷。

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

运行结果：

```
你好，世界
```

代码：

```cpp
#include <iostream>

int main() {
    std::cout << "你好\n世界";
    return 0;
}
```

程序运行时会在终端上显示 `你好[换行]世界`。此处字符串中的 `\n` 表示换行符，输出该字符后，终端上的文字会另起一行书写。

换行符 `\n` 是字符串字面量所支持的转义符的一种，详见 [string.literial.escape]。

> 如果真的要输出 `\n` 的话，需要写 `"你好，世界\\n"`，此处 `\\` 也是一种转义符，表示原原本本的 `\` 自己。

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

### 连续使用 `<<` 运算符

### 不推荐使用 `std::endl`

## 输入流

`std::cin` 是标准输入流，只读。对应于 C 语言的 `stdin` ^[[cstdio.stdio]]。

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

### 输入流重定向

在操作系统提供的命令提示符或 Shell ^[[os.shell]] 中，输入流可以被重定向 ^[[os.io.redirect]] 到指定文件。

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

> 操作系统实现重定向的机制和原理详见 [os.io.redirect]。

> 进一步，还可以利用操作系统的管道机制，把一个程序的输出流作为另一个程序的输入流，详见 [os.io.pipe]。

## 输出流

`std::cout` 是标准输出流，只写。对应于 C 语言的 `stdout`。

通常用于将文本打印到终端上，也可以被重定向到指定文件。

`std::clog` 是标准错误输出流，只写。对应于 C 语言的 `stderr`。

通通常用于将文本打印到终端上。常用于输出日志和调试信息。

`std::cerr` 是标准错误输出流，只写。对应于 C 语言的 `stderr`。

### 输出流的行缓冲机制 ^[[iostream.buf]]

```cpp
int main() {
    std::cout << "这是一条 cout 消息\n";
    std::cerr << "这是一条 cerr 消息\n";
}
```

### `std::cerr` 与 `std::clog` 的区别

`std::cerr` 无缓冲，每次 `<<` 后都会自动强制刷新缓冲区 ^[[iostream.buf]] (flush buffer)。

而 `std::clog` 不会强制刷新缓冲，需要手动 `std::clog << std::flush` 才能刷新。

由于所有流默认都开启了行缓冲，即检测到输出了换行符 `\n` 时会自动 flush 刷新缓冲区，因此只要你每次 clog 后都输出 `\n` 换行，那和 cerr 就没有区别，反而因为 `std::clog` 避免了多余的反复刷新，提升了性能。

这样设计的初衷是方便程序要打印的错误信息及时输出到终端上显示，让用户及时看到。

否则如果程序调用 clog 来输出，由于 clog 会把文本缓冲起来，稍后才一次性提交给操作系统，如果程序在缓冲尚未提交期间异常崩溃而不是正常退出 ^[[life.atexit]]，clog 缓冲的部分文本就会丢失。

关于输出流的缓冲与刷新机制，以及关闭输出缓冲的方法，参见 [iostream.buf]。

### 输出流重定向

在操作系统提供的命令提示符或 Shell ^[[os.shell]] 中，输入流可以被重定向 ^[[os.io.redirect]] 到指定文件。

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

用户无需在终端键入任何内容，程序自动从指定的输入文件 `file.txt` 读取了输入。

> 操作系统实现重定向的机制和原理详见 [os.io.redirect]。

> 进一步，还可以利用操作系统的管道机制，把一个程序的输出流作为另一个程序的输入流，详见 [os.io.pipe]。

### 输入流的行缓冲机制 ^[[os.terminal.buf]]

输入流的行缓冲指的是，由于操作系统的设计原因：用户输入的字符，并不会马上被 `std::cin` 收到，而是等用户按下回车键 (enter) 后才会把输入的文本发送给程序。

这样设计的初衷是为了让用户可以在输入一部分文本后，检查输入是否正确，按下回车键 (backspace) 删除输错的部分，然后再输入剩下的文本，确认无误后再按下回车键。

糟糕的是，输入流的行缓冲是操作系统层面实现的，用户很难绕开。因此，输入流的行缓冲与输出流的行缓冲有着很大不同。

- 输出流的缓冲机制是标准库实现的，可以通过调用 `std::ostream` 的接口关闭缓冲。

- 输入流的缓冲机制是操作系统实现的，需要调用操作系统的底层 API 才能关闭，例如利用 termios ^[os.terminal.termios]。

关于输入流缓冲机制的进一步详细说明，以及关闭输入缓冲的方法，参见 [os.terminal.buf]。对于 Windows 系统，可以用无缓冲的 getch 系列函数，参见 [win.crt.conio]。

## 不推荐的 `using namespace std`

你可能看到有些人的 C++ 代码中使用 `cout` 时不用加 `std::` 前缀。

这是因为他们在导入头文件后加入了一行 `using namespace std` ^[[namespace.using]]，使 `std::` 名字空间下的所有符号都暴露到了全局名字空间。

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
