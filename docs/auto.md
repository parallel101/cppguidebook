# `auto` 神教 (未完工)

## 变量 `auto`

## 返回类型 `auto`

C++11 引入的 `auto` 关键字可以用作函数的返回类型，但它只是一个“占位”，让我们得以后置返回类型，并没有多大作用，非常残废。

```cpp
auto f() -> int;
// 等价于：
int f();
```

> {{ icon.fun }} 闹了半天，还是要写返回类型，就只是挪到后面去好看一点……

> {{ icon.detail }} 当初引入后置返回类型实际的用途是 `auto f(int x) -> decltype(x * x) { return x * x; }` 这种情况，但很容易被接下来 C++14 引入的真正 `auto` 返回类型推导平替了。

但是 C++14 引入了函数**返回类型推导**，`auto` 才算真正意义上能用做函数返回类型，它会自动根据函数中的 `return` 表达式推导出函数的返回类型。

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

## 参数类型 `auto`

C++20 引入了**模板参数推导**，可以让我们在函数参数中也使用 `auto`。

在函数参数中也使用 `auto` 实际上等价于将该参数声明为模板参数，仅仅是一种更便捷的写法。

```cpp
void func(auto x) {
    std::cout << x;
}
// 等价于:
template <typename T>
void func(T x) {
    std::cout << x;
}

func(1); // 自动推导为调用 func<int>(1)
func(3.14); // 自动推导为调用 func<double>(3.14)
```

如果参数类型的 `auto` 带有如 `auto &` 这样的修饰，则实际上等价于相应模板函数的 `T &`。

```cpp
// 自动推导为常引用
void func(auto const &x) {
    std::cout << x;
}
// 等价于:
template <typename T>
void func(T const &x) {
    std::cout << x;
}

// 自动推导为万能引用
void func(auto &&x) {
    std::cout << x;
}
// 等价于:
template <typename T>
void func(T &&x) {
    std::cout << x;
}
```

### `auto` 在多态中的妙用

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

## `auto` 推导为引用

TODO: 继续介绍 `auto`, `auto const`, `auto &`, `auto const &`, `auto &&`, `decltype(auto)`, `auto *`, `auto const *`
