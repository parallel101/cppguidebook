# `auto` 神教

## `auto` 关键字的前世今生

TODO

## 变量声明为 `auto`

```cpp
int i = 0;
```

### 用 `auto` 声明万物的好处

#### 避免复读类型

> {{ icon.fun }} 人类的本质是复读机。

```cpp
QSlider *slider = new QSlider();
std::shared_ptr<Test> test = std::make_shared<Test>();
```

```cpp
auto slider = new QSlider();
auto test = std::make_shared<Test>();
```

TODO

#### 模板编程产生的超长类型名喧宾夺主

在 C++98 时代，仅仅只是保存个迭代器作为变量，就得写一长串：

```cpp
std::map<std::string, int> tab;
std::map<std::string, int>::iterator it = tab.find("key");
```

这踏码的类型名比右侧的表达式都长了！

> {{ icon.fun }} 哮点解析：张心欣的第三条腿比另外两条腿都长。

有了 `auto` 以后，无需复读类型名和繁琐的 `::iterator` 废话，自动从右侧 `find` 函数的返回值推导出正确的类型。

```cpp
std::map<std::string, int> tab;
auto it = tab.find("key");
```

#### 避免未初始化

因为 `auto` 规定必须右侧有赋初始值（否则无法推导类型）。

所以只要你的代码规范能一直使用 `auto` 的话，就可以避免未初始化。

众所周知，读取一个未初始化的变量是未定义行为，C/C++ 程序员饱受其苦，小彭老师也好几次因为忘记初始化成员指针。

例如，你平时可能一不小心写：

```cpp
int i;
cout << i; // 未定义行为！此时 i 还没有初始化
```

但是如果你用了 `auto`，那么 `auto i` 就会直接报错，提醒你没有赋初始值：

```cpp
auto i;  // 编译出错，强制提醒你必须赋初始值！
cout << i;
```

你意识到自己漏写了 `= 0`！于是你写上了初始值，编译才能通过。

```cpp
auto i = 0;
cout << i;
```

可见，只要你养成“总是 `auto`”的好习惯，就绝对不会忘记变量未初始化，因为 `auto` 会强制要求有初始值。

#### 自动适配类型，避免类型隐式转换

假设你有一个能返回 `int` 的函数：

```cpp
int getNum();
```

有多处使用了这个函数：

```cpp
int a = getNum();
...
int b = getNum() + 1;
...
```

假如你哪天遇到牢板需求改变，它说现在我们的 `Num` 需要是浮点数了！

```cpp
float getNum();
```

哎呀，你需要把之前那些“多处使用”里写的 `int` 全部一个个改成 `float`！

```cpp
float a = getNum();
...
float b = getNum() + 1;
...
```

如果漏改一个的话，就会发生隐式转换，并且只是警告，不会报错，你根本注意不到，精度就丢失了！

现在“马后炮”一下，如果当时你的“多处使用”用的是 `auto`，那该多好！自动适应！

```cpp
auto a = getNum();
...
auto b = getNum() + 1;
...
```

无论你今天 `getNum` 想返回 `float` 还是 `double`，只需要修改 `getNum` 的返回值一处，所有调用了 `getNum` 的地方都会自动适配！

> {{ icon.fun }} 专治张心欣这种小计级扒皮牢板骚动反复跳脚的情况，无需你一个个去狼狈的改来改回，一处修改，处处生效。

#### 统一写法，更可读

```cpp
std::vector<int> aVeryLongName(5);
```

```cpp
auto aVeryLongName = std::vector<int>(5);
```

TODO

#### 强制写明字面量类型，避免隐式转换

有同学反映，他想要创建一个 `size_t` 类型的整数，初始化为 3。

```cpp
size_t i = 3;  // 3 是 int 类型，这里初始化时发生了隐式转换，int 转为了 size_t
i = 0xffffffffff; // OK，在 size_t 范围内（64 位编译器）
```

如果直接改用 `auto` 的话，因为 `3` 这个字面量是 `int` 类型的，所以初始化出来的 `auto i` 也会被推导成 `int i`！

虽然目前初始只用到了 `3`，然而这位同学后面可能会用到 `size_t` 范围的更大整数存入，就存不下了。

```cpp
auto i = 3; // 错误！auto 会推导为 int 了！
i = 0xffffffffff; // 超出 int 范围！
```

由于 C++ 是静态编译，变量类型一旦确定就无法更改，我们必须在定义时就指定号范围更大的 `size_t`。

为了让 `auto` 推导出这位同学想要的 `size_t` 类型，我们可以在 `3` 这个字面量周围显式写出类型转换，将其转换为 `size_t`。

> {{ icon.tip }} 显式类型转换总比隐式的要好！

```
auto i = (size_t)3; // 正确
```

这里的类型转换用的是 C 语言的强制类型转换语法 `(size_t)3`，更好的写法是用括号包裹的 C++ 构造函数风格的强制类型转换语法：

```
auto i = size_t(3); // 正确
```

看起来就和调用了 `size_t` 的“构造函数”一样。这也符合我们前面说的统一写法，类型统一和值写在一起，以括号结合，更可读。

> {{ icon.detail }} 顺便一提，`0xffffffffff` 会是 `long` (Linux) 或 `long long` (Windows) 类型字面量，因为它已经超出了 `int` 范围，所以实际上 `auto i = 0xffffffffff` 会推导为 `long i`。字面量类型的规则是，如果还在 `int` 范围内（0x7fffffff 以内），那这个字面量就是 `int`；如果超过了 0x7fffffff 但不超过 0xffffffff，就会变成 `unsigned int`；如果超过了 0xffffffff 就会自动变成 `long` (Linux) 或 `long long` (Windows) ；超过 0x7fffffffffffffff 则变成 `unsigned long` (Linux) 或 `unsigned long long` (Windows) ——这时和手动加 `ULL` 等后缀等价，无后缀时默认 `int`，如果超过了 `int` 编译器会自动推测一个最合适的。

如果需要其他类型的变量，改用 `short(3)`，`uint8_t(3)` 配合 `auto` 不就行了，根本没必要把类型前置。

#### 避免语法歧义

TODO

### `auto` 的小插曲：初始化列表

TODO

## 返回类型 `auto`

C++11 引入的 `auto` 关键字可以用作函数的返回类型，但它只是一个“占位”，让我们得以后置返回类型，并没有多大作用，所以 C++11 这版的 `auto` 非常残废。

```cpp
auto f() -> int;
// 等价于：
int f();
```

> {{ icon.fun }} 闹了半天，还是要写返回类型，就只是挪到后面去好看一点……

> {{ icon.detail }} 当初引入后置返回类型实际的用途是 `auto f(int x) -> decltype(x * x) { return x * x; }` 这种情况，但很容易被接下来 C++14 引入的真正 `auto` 返回类型推导平替了。

终于，C++14 引入了函数**返回类型推导**，`auto` 才算真正意义上能用做函数返回类型！它会自动根据函数中的 `return` 表达式推导出函数的返回类型。

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

### 返回引用类型

返回类型声明为 `auto`，可以自动推导返回类型，但总是推导出普通的值类型，绝对不会带有引用或 `const` 修饰。

如果需要返回一个引用，并且希望自动推导引用的类型，可以写 `auto &`。

```cpp
int i;
int &ref = i;

auto f() { // 返回类型推导为 int
    return i;
}

auto f() { // 返回类型推导依然为 int
    return ref;
}

auto &f() { // 返回类型这才能推导为 int &
    return ref;
}

auto &f() { // 编译期报错：1 是纯右值，不可转为左值引用
    return 1;
}

auto &f() { // 运行时出错：空悬引用是未定义行为
    int local = 42;
    return local;
}
```

这里的 `auto` 还可以带有 `const` 修饰，例如 `auto const &` 可以让返回类型变成带有 `const` 修饰的常引用。

```cpp
int i;
int &ref = i;

```cpp
int i;

auto getValue() { // 返回类型推导为 int
    return i;
}

auto &getRef() { // 返回类型推导为 int &
    return i;
}

auto const &getConstRef() { // 返回类型推导为 int const &
    return i;
}
```

> {{ icon.tip }} `auto const &` 与 `const auto &` 完全等价，只是代码习惯问题。

有趣的是，如果 `i` 是 `int const` 类型，则 `auto &` 也可以自动推导为 `int const &` 且不报错。

```cpp
const int i;

auto const &getConstRef() { // 返回类型推导为 int const &
    return i;
}

auto &getRef() { // 返回类型也会被推导为 int const &
    return i;
}

int &getRef() { // 报错！
    return i;
}
```

> {{ icon.tip }} `int const` 与 `const int` 是完全等价的，只是代码习惯问题。

> {{ icon.detail }} `auto &` 可以兼容 `int const &`，而 `int &` 就不能兼容 `int const &`！很奇怪吧？这是因为 `auto` 不一定必须是 `int`，也可以是 `const int` 这一整个类型。你可以把 `auto` 看作和模板函数参数一样，模板函数参数的 `T &` 一样可以通过将 `T = const int` 从而捕获 `const int &`。

如果要允许 `auto` 推导为右值引用，只需写 `auto &&`。

```cpp
std::string str;

auto &&getRVRef() { // std::string &&
    return std::move(str);
}

auto &getRef() { // std::string &
    return str;
}

auto const &getConstRef() { // std::string const &
    return str;
}
```

正如 `auto &` 可以兼容 `auto const &` 一样，由于 C++ 的某些特色机制，`auto &&` 其实也可以兼容 `auto &`！

所以 `auto &&` 实际上不止支持右值引用，也支持左值引用，因此被称为“万能引用”。

也就是说，其实我们可以都写作 `auto &&`！让编译器自动根据我们 `return` 语句的表达式类型，判断返回类型是左还是右引用。

```cpp
std::string str;

auto &&getRVRef() { // std::string &&
    return std::move(str);
}

auto &&getRef() { // std::string &
    return str;
}

auto const &getConstRef() { // std::string const &
    return str;
}
```

`auto &&` 不仅能推导为右值引用，也能推导为左值引用，常左值引用。

可以理解为集合的包含关系：`auto &&` > `auto &` > `auto const &`

所以 `auto &&` 实际上可以推导所有引用，不论左右。

> {{ icon.detail }} 这里的原因和刚才 `auto = int const` 从而 `auto &` 可以接纳 `int const &` 一样，`auto &&` 可以接纳 `int &` 是因为 C++ 特色的“引用折叠”机制：`& && = &` 即左引用碰到右引用，会得到左引用。所以编译器可以通过令 `auto = int &` 从而使得 `auto && = int & && = int &`，从而实际上 `auto &&` 看似是右值引用，但是因为可以给 `auto` 带入一个左值引用 `int &`，然后让左引用 `&` 与右引用 `&&` “湮灭”，最终只剩下一个左引用 `&`，在之后的模板函数专题中会更详细介绍这一特色机制。

这就是为什么 `int &&` 就只是右值引用，而 `auto &&` 以及 `T &&` 则会叫做万能引用。一旦允许前面的参数为 `auto` 或者模板参数，就可以代换，就可以实现左右通吃。

### 真正的万能 `decltype(auto)`

返回类型声明为 `decltype(auto)` 的效果等价于把返回类型替换为 `decltype((返回表达式))`：

```cpp
int i;

decltype(auto) func() {
    return i;
}
// 等价于：
decltype((i)) func() {
    return i;
}
// 等价于：
int &func() {
    return i;
}
```

> {{ icon.warn }} 注意 `decltype(i)` 是 `int` 而 `decltype((i))` 是 `int &`。这是因为 `decltype` 实际上有两个版本！当 `decltype` 中的内容只是单独的一个标识符（变量名）时，会得到变量定义时的类型；而当 `decltype` 中的内容不是单纯的变量名，而是一个复杂的表达式时，就会进入 `decltype` 的第二个版本：表达式版，会求表达式的类型，例如当变量为 `int` 时，表达式 `(i)` 的类型是左值引用，`int &`，而变量本身 `i` 的类型则是 `int`。此处加上 `()` 就是为了让 `decltype` 被迫进入“表达式”的那个版本，`decltype(auto)` 遵循的也是“表达式”这个版本的结果。

```cpp
int i;

decltype(auto) func() {
    return i;
}
// 等价于：
decltype((i + 1)) func() {
    return i + 1;
}
// 等价于：
int func() {
    return i + 1;
}
```

```cpp
int i;

decltype(auto) func() {
    return std::move(i);
}
// 等价于：
decltype((std::move(i))) func() {
    return std::move(i);
}
// 等价于：
int &&func() {
    return std::move(i);
}
```

以上介绍的这些引用推导规则，其实也适用于局部变量的 `auto`，例如：

```cpp
auto i = 0;              // int i = 0
auto &ref = i;           // int &ref = i
auto const &cref = i;    // int const &cref = i
auto &&rvref = std::move(i); // int &&rvref = move(i)

decltype(auto) j = i;    // int j = i
decltype(auto) k = ref;  // int &k = ref
decltype(auto) l = cref; // int const &l = cref
decltype(auto) m = std::move(rvref); // int &&m = rvref
```

## 范围 for 循环中的 `auto &`

众所周知，在 C++11 的“范围 for 循环” (range-based for loop) 语法中，`auto` 的出镜率很高。

但是如果只是写 `auto i: arr` 的话，这会从 arr 中拷贝一份新的 `i` 变量出来，不仅产生了额外的开销，还意味着你对这 `i` 变量的修改不会反映到 `arr` 中原本的元素中去。

```cpp
std::vector<int> arr = {1, 2, 3};
for (auto i: arr) {  // auto i 推导为 int i，会拷贝一份新的 int 变量
    i += 1; // 错误的写法，这样只是修改了 int 变量
}
print(arr); // 依然是 {1, 2, 3}
```

更好的写法是 `auto &i: arr`，保存一份对数组中元素的引用，不仅避免了拷贝的开销（如果不是 `int` 而是其他更大的类型的话，这是一笔不小的开销），而且允许你就地修改数组中元素的值。

```cpp
std::vector<int> arr = {1, 2, 3};
for (auto &i: arr) {  // auto &i 推导为 int &i，保存的是对 arr 中原元素的一份引用，不发生拷贝
    i += 1; // 因为 i 现在是对 arr 中真正元素的引用，对其修改也会成功反映到原 arr 中去
}
print(arr); // 变成了 {2, 3, 4}
```

如果不打算修改数组，也可以用 `auto const &`，让捕获到的引用添加上 `const` 修饰，避免一不小心修改了数组，同时提升代码可读性（人家一看就懂哪些 for 循环是想要修改原值，哪些不会修改原值）。

```cpp
std::vector<int> arr = {1, 2, 3};
for (auto const &i: arr) {  // auto const &i 推导为 int const &i，保存的是对 arr 中原元素的一份常引用，不发生拷贝，且不可修改
    i += 1; // 编译期出错！const 引用不可修改
}
```

> {{ icon.tip }} 对于遍历 `std::map`，由于刚才提到的 `auto &` 实际上也兼容常引用，而 map 的值类型是 `std::pair<const K, V>`，所以即使你只需修改 `V` 的部分，只需使用 `auto &` 配合 C++17 的“结构化绑定” (structural-binding) 语法拆包即可，`K` 的部分会自动带上 `const`，不会出现编译错误的。

```cpp
std::map<std::string, std::string> table;
for (auto &[k, v]: table) { // 编译通过：k 的部分会自动带上 const
    k = "hello"; // 编译出错：k 推导为 std::string const & 不可修改
    v = "world"; // 没问题：v 推导为 std::string & 可以就地修改
}
```

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

实际上等价于模板函数的如下写法：

```cpp
template <class T>
decltype(T() * T()) square(T x) {
    return x * x;
}
```

### 参数 `auto` 推导为引用

和之前变量 `auto`，返回类型 `auto` 的 `auto &`、`auto const &`、`auto &&` 大差不差，C++20 这个参数 `auto` 同样也支持推导为引用。

```cpp
void passByValue(auto x) { // 参数类型推导为 int
    x = 42;
}

void passByRef(auto &x) { // 参数类型推导为 int &
    x = 42;
}

void passByConstRef(auto const &x) { // 参数类型推导为 int const &
    x = 42; // 编译期错误：常引用无法写入！
}

int x = 1;
passByValue(x);
cout << x; // 还是 1
passByRef(x);
cout << x; // 42
```

```cpp
void passByRef(auto &x) {
    x = 1;
}

int x = 1;
const int const_x = 1;
passByRef(i); // 参数类型推导为 int &
passByRef(const_x); // 参数类型推导为 const int &
```

由于 `auto &` 兼容 `auto const &` 的尿性，此处第二个调用 `passByRef` 会把参数类型推导为 `const int &`，这会导致里面的 x = 42 编译出错！

- 所以 `auto &` 实际上也允许传入 `const` 变量的引用，非常恼人，不要掉以轻心。
- 而 `auto const &` 则可以安心，一定是带 `const` 的。

> {{ icon.fun }} 所以实际上最常用的是 `auto const &`。

不仅如此 `auto const &` 参数还可以传入纯右值（利用了 C++ 可以自动把纯右值转为 `const` 左引用的特性）。

对于已有的变量传入，可以避免一次拷贝；对于就地创建的纯右值表达式，则自动转换，非常方便。

```cpp
void passByConstRef(auto const &cref) {
    std::cout << cref;
}

int i = 42;
passByConstRef(i);  // 传入 i 的引用
passByConstRef(42); // 利用 C++ 自动把纯右值 “42” 自动转为 const 左值的特性
```

对于这种自动转出来的 `const` 左值引用，其实际上是在栈上自动创建了一个 `const` 变量保存你临时创建的参数，然后在当前行结束后自动析构。

```cpp
passByConstRef(42);
// 等价于：
{
    const int tmp = 42;
    passByConstRef(tmp); // 传入的是这个自动生成 tmp 变量的 const 引用
}
```

这个自动生成的 `tmp` 变量的生命周期是“一条语句”，也就是当前分号结束前，该变量的生命周期都存在，直到分号结束后才会析构，所以如下代码是安全的：

```cpp
void someCFunc(const char *name);

someCFunc(std::string("hello").c_str());
```

> {{ icon.detail }} 此处 `std::string("hello")` 构造出的临时 `string` 类型变量的生命周期直到 `;` 才结束，而这时 `someCFunc` 早已执行完毕返回了，只要 `someCFunc` 对 `name` 的访问集中在当前这次函数调用中，没有把 `name` 参数存到全局变量中去，就不会有任何空悬指针问题。

### `auto &&` 参数万能引用及其转发

TODO

然而，由于 C++ “默认自动变左值”的糟糕特色，即使你将一个传入时是右值的引用直接转发给另一个函数，这个参数也会默默退化成左值类型，需要再 `std::move` 一次才能保持他一直处于右值类型。

### `std::forward` 帮手函数介绍
