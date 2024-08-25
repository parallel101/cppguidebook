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

## `auto` 神教

### 变量 `auto`

### 返回类型 `auto`

C++11 引入的 `auto` 关键字可以用作函数的返回类型，但它只是一个“占位”，让我们得以后置返回类型，并没有多大作用，非常残废。

```cpp
auto f() -> int;
// 等价于：
int f();
```

> {{ icon.fun }} 闹了半天，还是要写返回类型，就只是挪到后面去好看一点……

> {{ icon.detail }} 当初引入后置返回类型实际的用途是 `auto f(int x) -> decltype(x * x) { return x * x; }` 这种情况，但很容易被接下来 C++14 引入的真正 `auto` 返回类型推导平替了。

C++14 引入了函数**返回类型推导**，`auto` 才算真正意义上能用做函数返回类型，它会自动根据函数中的 `return` 表达式推导出函数的返回类型。

```cpp
auto f(int x) {
  return x * x;  // 表达式 `x * x` 的类型为 int，所以 auto 类型推导为 int
}
// 等价于：
int f() {
  return x * x;
}
```

如果函数中没有 `return` 语句，那么 `auto` 会被自动推导为 `void`，非常方便。

```cpp
auto f() {
  std::println("hello");
}
// 等价于：
void f() {
  std::println("hello");
}
```

值得注意的是，返回类型用 `auto` 来推导的函数，如果有多条 `return` 语句，那么他们必须都返回相同的类型，否则报错。

```cpp
auto f(int x) {
  if (x > 0) {
    return 1;    // int
  } else {
    return 3.14; // double
  }
} // 错误：有歧义，无法确定 auto 应该推导为 int 还是 double
```

`auto` 还有一个缺点是，无法用于“分离声明和定义”的情况。因为推导 `auto` 类型需要知道函数体，才能看到里面的 `return` 表达式是什么类型。所以当 `auto` 返回类型被用于函数的非定义声明时，会直接报错。

```cpp
auto f();  // 错误：看不到函数体，无法推导返回类型

auto f() { // 编译通过：auto 推导为 int
    return 1;  // 1 是 int 类型的表达式
}
```

因此，`auto` 通常只适用于头文件中“就地定义”的 `inline` 函数，不适合需要“分离 .cpp 文件”的函数。

### 参数类型 `auto`

C++20 引入了**模板参数推导**，可以让我们在函数参数中也使用 `auto`。

TODO: 介绍

传统的，基于类型重载的：

```cpp
int square(int x) {
    return x * x;
}

double square(double x) {
    return x * x;
}

int main() {
    square(2);    // 4（调用 int 版重载）
    square(3.14); // 9.8596（调用 double 版重载）
    // 如果现在又需要 float 版呢？又得写一版重载，内容还是完全一样的，浪费时间
}
```

基于 `auto` 模板参数推导的：

```cpp
auto square(auto x) {
    return x * x;
}

int main() {
    square(2);    // 4（auto 推导为 int）
    square(3.14); // 9.8596（auto 推导为 double）
    // 即使未来产生了 float 版的需求，也不用添加任何代码，因为是 square 是很方便的模板函数
}
```

### `auto` 推导为引用

TODO: 继续介绍 `auto &`, `auto const &`, `auto &&`, `decltype(auto)`, `auto *`, `auto const *`
