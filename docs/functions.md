# 认识函数 (未完工)

[TOC]

# 自定义函数

```cpp
int square(int x) {
}
```

## 调用函数

```cpp
TODO: println 参数演示
```

## 函数的返回值

函数可以没有返回值，只需要声明函数时返回类型声明为 `void` 即可，调用这样的函数只是为了他的副作用（如修改全局变量，输出文本到控制台，修改引用参数等）。

```cpp
void compute()
{
    return;
}
```

> {{ icon.tip }} 对于没有返回值（返回类型为 `void`）的函数，可以省略 `return` 不写。

```cpp
void compute()
{
    // 没问题
}
```

> {{ icon.warn }} 对于返回类型不为 `void` 的函数，必须写 `return` 语句，如果漏写，会出现可怕的未定义行为 (undefined behaviour)。编译器不一定会报错，而是到运行时才出现崩溃等现象。建议 GCC 用户开启 `-Werror=return-type` 让编译器在编译时就检测此类错误，MSVC 则是开启 `/we4716`。更多未定义行为可以看我们的[未定义行为列表](undef.md)章节。
> {{ icon.detail }} 但有两个例外：1. main 函数是特殊的可以不写 return 语句，默认会自动帮你 `return 0;`。2. 具有 co_return 或 co_await 的协程函数可以不写 return 语句。

### 接住返回值

## 函数的参数

### 形参 vs 实参

### 按引用传参 vs 按值传参

TODO：和 Python、Java 对比

### C 风格变长参数

## 模板函数

TODO：更多介绍函数

## main 函数的参数

TODO
