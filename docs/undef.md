# 未定义行为完整列表

[TOC]

如有疏漏，可以在 [GitHub](https://github.com/parallel101/cppguidebook) 补充。

## 建议开启标准库的调试模式

可以帮助你监测未定义行为

- msvc: Debug 配置
- gcc: 定义 `_GLIBCXX_DEBUG` 宏

## 空指针类

### 不能解引用空指针（通常会产生崩溃，但也可能被优化产生奇怪的现象）

只要解引用就错了，无论是否读取或写入

```cpp
int *p = nullptr;
*p;          // 错！
&*p;         // 错！
*p = 0;      // 错！
int i = *p;  // 错！
```

```cpp
unique_ptr<int> p = nullptr;
p.get();     // 可以
&*p;         // 错！
```

例如在 Debug 配置的 MSVC STL 中，`&*p` 会产生断言异常，而 `p.get()` 不会。

```cpp
if (&*p != nullptr) { // 可能被优化为 if (1)，因为未定义行为被排除了
}
if (p != nullptr) {   // 不会被优化，正常判断
}
```

### 不能解引用 end 迭代器

```cpp
std::vector<int> v = {1, 2, 3, 4};
int *begin = &*v.begin();
int *end = &*v.end(); // 错！
```

```cpp
std::vector<int> v = {};
int *begin = &*v.begin(); // 错!
int *end = &*v.end();     // 错！
```

建议改用 data 和 size

```cpp
std::vector<int> v = {1, 2, 3, 4};
int *begin = v.data();
int *end = v.data() + v.size();
```

### this 指针不能为空

```cpp
struct C {
    void print() {
        if (this == nullptr) { // 此分支可能会被优化为 if (0) { ... } 从而永不生效
            std::cout << "this 是空\n";
        }
    }
};

void func() {
    C *c = nullptr;
    c->print(); // 错！
}
```

### 空指针不能调用成员函数

```cpp
struct C{
    void f() {}
    static void f2() {}
};

void func(){
    C* c = nullptr;
    c->f();  // 行为未定义
    c->f2(); // 行为未定义
}
```

本质上是因为**空指针解引用**。对于内建类型，表达式 `E1->E2` 与 `(*E1).E2` 严格等价，任何指针类型都是内建类型。

`c->f()`、`c->f2()` 等价于：

```cpp
(*c).f();
(*c).f2();
```

## 指针别名类

### reinterpret_cast 后以不兼容的类型访问

```cpp
int i;
float f = *(float *)&i; // 错！
```

例外：char、signed char、unsigned char 和 std::byte 总是兼容任何类型

```cpp
int i;
char *buf = (char *)&i; // 可以
buf[0] = 1;             // 可以
```

> uint8_t 是 unsigned char 的别名，所以也兼容任何类型

例外：int 和 unsigned int 互相兼容

```cpp
int i;
unsigned int f = *(unsigned int *)&i; // 可以
```

例外：const int * 和 int * 互相兼容（二级指针强转）

```cpp
const int *cp;
int *p = *(int **)&cp;  // 可以
```

注意：只取决于访问时的类型是否正确，中间可以转换为别的类型（如 void * 和 uintptr_t），只需最后访问时转换回正确的指针类型即可

```cpp
int i;
*(int *)(uintptr_t)&i;  // 可以
*(int *)(void *)&i;  // 可以
*(int *)(float *)&i;  // 可以
```

### union 访问不是激活的成员

```cpp
float bitCast(int i) {
    union {
        int i;
        float f;
    } u;
    u.i = i;
    return u.f; // 错！
}
```

特例：公共的前缀成员可以安全地访问

```cpp
int foo(int i) {
    union {
        struct {
            int tag;
            int value;
        } m1;
        struct {
            int tag;
            float value;
        } m2;
    } u;
    u.m1.tag = i;
    return u.m2.tag; // 可以
}
```

如需在 float 和 int 之间按位转换，建议改用 memcpy，因为 memcpy 内部被认为是以 char 指针访问的，char 总是兼容任何类型

```cpp
float bitCast(int i) {
    float f;
    memcpy(&f, &i, sizeof(i));
    return f;
}
```

或 C++20 的 `std::bit_cast`

```cpp
float bitCast(int i) {
    float f = std::bit_cast<float>(i);
    return f;
}
```

### T 类型指针必须对齐到 alignof(T)

```cpp
struct alignas(64) C { // 假设 alignof(int) 是 4
    int i;
    char c;
};

C *p = (C *)malloc(sizeof(C)); // 错！malloc 产生的指针只保证对齐到 max_align_t（GCC 上是 16 字节）大小，并不保证对齐到 C 所需的 64 字节
C *p = new C;  // 可以，new T 总是保证对齐到 alignof(T)
```

```cpp
char buf[sizeof(int)];
int *p = (int *)buf;  // 错！
```

```cpp
alignas(alignof(int)) char buf[sizeof(int)];
int *p = (int *)buf;  // 可以
```

```cpp
char buf[sizeof(int) * 2];
int *p = (int *)(((uintptr_t)buf + sizeof(int) - 1) & ~(alignof(int) - 1));  // 可以
```

### 从父类 static_cast 到不符合的子类后访问

```cpp
struct Base {};
struct Derived : Base {};

Base b;
Derived d = *(Derived *)&b;              // 错！
Derived d = *static_cast<Derived *>(&b); // 错！
Derived d = static_cast<Derived &>(b);   // 错！
```

```cpp
Derived obj;
Base *bp = &obj;
Derived d = *(Derived *)bp;              // 可以
Derived d = *static_cast<Derived *>(bp); // 可以
Derived d = static_cast<Derived &>(*bp); // 可以
```

### bool 类型不得出现 0 和 1 以外的值

布尔类型 bool，只有 true 和 false 两种取值。

bool 占据 1 字节（8 位）内存空间，其中有效的位只有最低位。这个最低位可以是 0 或 1，但其余 7 位仍然参与 bool 的值表示，而且必须始终保持为 0。

如果其余位中出现了非 0 的位，也就是出现 0 和 1 以外的取值，则读取该 bool 值或者通过 std::bit_cast 产生这种值是未定义行为。

```cpp
bool b0;
std::memset(&b0, 1, 0);
if (b0) { /* ... */ } else { /* ... */ } // 可以，b0 == false
auto b1 = std::bit_cast<bool>(char(0));  // 可以，b1 == false
```

```cpp
bool b2;
std::memset(&b2, 1, 1);
if (b2) { /* ... */ } else { /* ... */ } // 可以，b2 == true
auto b3 = std::bit_cast<bool>(char(1));  // 可以，b3 == true
```

```cpp
bool b4;
std::memset(&b4, 1, 2);
if (b4) { /* ... */ } else { /* ... */ } // 未定义行为
auto b5 = std::bit_cast<bool>(char(2));  // 未定义行为
```

## 算数类

### 有符号整数的加减乘除模不能溢出

```cpp
int i = INT_MAX;
i + 1;  // 错！
```

但无符号可以，无符号整数保证：溢出必定回环 (wrap-around)

```cpp
unsigned int i = UINT_MAX;
i + 1;  // 可以，会得到 0
```

如需对有符号整数做回环，可以先转换为相应的 unsigned 类型，算完后再转回来

```cpp
int i = INT_MAX;
(int)((unsigned int)i + 1);  // 可以，会得到一个负数 INT_MIN
```

> {{ icon.detail }} 如下写法更具有可移植性，因为无符号数向有符号数转型时若超出有符号数的表示范围则为实现定义行为（编译器厂商决定结果，但不是未定义行为）

```cpp
std::bit_cast<int>((unsigned int)i + i);
```

有符号整数的加减乘除模运算结果结果必须在表示范围内：例如对于 int a 和 int b，若 a/b 的结果不可用 int 表示，那么 a/b 和 a%b 均未定义

```cpp
INT_MIN % -1; // 错！
INT_MIN / -1; // 错！
```

### 左移或右移的位数，不得超过整数类型上限，不得为负

```cpp
unsigned int i = 0;
i << 31;  // 可以
i << 32;  // 错！
i << 0;   // 可以
i << -1;  // 错！
```

但是你还需要考虑一件事情：**隐式转换**，或者直接点说：**整数提升**。

- 在 C++ 中算术运算符不接受小于 int 的类型进行运算。如果你觉得可以，那只是隐式转换，整形提升了。

```cpp
std::uint8_t c{ '0' };
using T1 = decltype(c << 1); // int
```

即使移位大于等于 8 也不成问题。

---

对于有符号整数，左移还不得破坏符号位

```cpp
int i = 0;
i << 1;   // 可以
i << 31;  // 错！
unsigned int u = 0;
u << 31; // 可以
```

如需处理来自用户输入的位移数量，可以先做范围检测

```cpp
int shift;
cin >> shift;

unsigned int u = 0;
int i = 0;
(shift > 0 && shift < 32) ? (u << shift) : 0; // 可以
(shift > 0 && shift < 31) ? (i << shift) : 0; // 可以
```

### 除数不能为 0

```cpp
int i = 42;
int j = 0;
i / j;  // 错！
i % j;  // 错！
```

## 求值顺序类

### 同一表达式内，对同一个变量有多个自增/自减运算

```cpp
int i = 5;
int j = (++i) + (++i);    // j 的值未定义
```

```cpp
int i = 5;
int a[10] = {};
int j = a[i++] + a[i++];  // j 的值未定义
```

```cpp
int i = 5;
int j = (++i) + i;        // j 的值未定义
```

```cpp
int i1 = 5;
int i2 = 5;
int j = (++i1) + (++i2); // 正确，j 会得到 12
```

> {{ icon.fun }} 转发给你身边的谭浩强受害者看（`i+++++i`）。

### 内建类型的二元运算符，其左右两个参数求值的顺序是不确定的

在标准看来，+ 运算符两侧是“同时”求值的，即“interleaved”，实际执行顺序并不确定。

对于 a + b，我们不能假定总是左侧表达式 a 先求值。

不过，虽然运算符两个参数的求值顺序“未指定(unspecified)”，但并不是“未定义(undefined)”。

> 但左右两侧涉及自增/自减运算符的情况仍然是未定义行为。

```cpp
int f1() {
    printf("f1\n");
    return 1;
}

int f2() {
    printf("f2\n");
    return 2;
}

int j = f1() + f2();   // 可能打印 f1 f2，也可能打印 f2 f1，但 j 最终的结果一定是 3
```

未指定和未定义是不同的！有未定义行为的程序是非法(ill-formed)的，但未指定只是会让结果无法确定，但一定能正常运行：要么 f1 先运行，要么 f2 先运行。

### 函数参数求值的顺序是不确定的

```cpp
int f1() {
    printf("f1\n");
    return 1;
}

int f2() {
    printf("f2\n");
    return 2;
}

void foo(int i, int j) {
    printf("%d %d\n", i, j);
}

foo(f1(), f2());   // 可能打印 f1 f2 1 2，也可能打印 f2 f1 1 2
```

代码中，f1 和 f2 的求值顺序虽然未指定，但可以保证 foo 函数体一定在执行完毕后才会开始。

同一条语句中所有子表达式的执行顺序就像一颗树，树中两个子节点执行顺序是不确定的；但可以肯定的是，树的子节点一定先于他们的父节点执行。

同样地，这只是未指定(unspecified)行为而不是未定义(undefined)行为，结果必然是 f1 f2 或 f2 f1 两种可能之一，不会让程序出现未定义值的情况。

注意，求值顺序未指定仅限同一语句（“同一行”）内，对于互相独立的多条语句，依然是有强先后顺序的。

```cpp
int f1() {
    printf("f1\n");
    return 1;
}

int f2() {
    printf("f2\n");
    return 2;
}

void foo(int i, int j) {
}

foo(f1(), f2());  // 可能打印 f1 f2，也可能打印 f2 f1

f1(); f2();       // 必然打印 f1 f2
```

不过，涉及自增的话，就还是未定义行为，而不是未指定了。

```cpp
int i = 5;
foo(i++, i++);   // 会打印出什么？未定义行为
```

```cpp
int i = 5;
int j = 5;
foo(i++, j++);   // 必然打印出 5 5
```

## 函数类

### 返回类型不为 void 的函数，必须有 return 语句

```cpp
int func() {
    int i = 42;
    // 错！会导致 func 返回时程序崩溃，且编译器只是警告，不报错
}

int func() {
    int i = 42;
    return i;  // 正确
}

void func() {
    int i = 42;
    // 返回 void 的函数，return 语句可以省略
}
```

坑人之处在于，忘记写，不会报错，编译器只是警告。

为了避免忘记写 return 语句，建议 gcc 编译器开启 `-Werror=return-type` 选项，将不写返回语句的警告转化为错误

注意，在有分支的非 void 函数中，必须所有可达分支都有 return 语句

```cpp
int func(int x) {
    if (x < 0)
        return -x;
    if (x > 0)
        return x;
    // 如果调用了 func(0)，那么会抵达没有 return 的分支，触发未定义行为
}
```

> {{ icon.detail }} 没有 return 的分支相当于写了一个 std::unreachable()

但也有例外：

1. 主函数 `main` 可以不写 `return` 语句，默认自带 `return 0;`
2. 协程函数可以不写 `return` 语句，如果有 `co_return` 或者协程返回类型为 `void` 且具有至少一个 `co_await` 出现

### 函数指针被调用时，不能为空

```cpp
typedef void (*func_t)();

func_t func = nullptr;
func();    // 错！
```

《经典再现》

```cpp
#include <cstdio>

static void func() {
    printf("func called\n");
}

typedef void (*func_t)();

static func_t fp = nullptr;

extern void set_fp() { // 导出符号，虽然没人调用，却影响了 clang 的优化决策
    fp = func;
}

int main() {
    fp(); // Release 时，clang 会把这一行直接优化成 func()
    return 0;
}
```

### 函数指针被调用时，参数列表或返回值必须匹配

```cpp
void f1(int *p) {
    printf("f1(%p)", p);
}

void (*fp)(const int *);
fp = (void (*)(const int *)) f1;  // 错误

int i;
fp = (void (*)(const int *)) &i;  // 错误
```

### 普通函数指针与成员函数指针不能互转

```cpp
struct Class {
    void mf() {
        printf("成员函数\n");
    }
};

union {
    void (Class::*member_func)();
    void (*free_func)(Class *);
} u;
u.member_func = &Class::mf;
Class c;
u.free_func(&c); // 错误
```

## 生命周期类

### 不能读取未初始化的变量

```cpp
int i;
cout << i; // 错！

int i = 0;
cout << i; // 可以，会读到 0

int arr[10];
cout << arr[0]; // 错！

int arr[10] = {};
cout << arr[0]; // 可以，会读到 0
```

### 指针的加减法不能超越数组边界

```cpp
int arr[10];
int *p = &arr[0];
p + 1;     // 可以
p + 10;    // 可以
p + 11;    // 错！
```

### 可以有指向数组尾部的指针（类似 end 迭代器），但不能解引用

```cpp
int arr[10];
int *p = &arr[0];
int *end = p + 10; // 可以
*end;              // 错！
```

### 不能访问未初始化的指针

```cpp
int *p;
*p; // 错！
```

```cpp
struct Dog {
    int age;
};

struct Person {
    Dog *dog;
};

Person *p = new Person;
cout << p->dog->age; // 错！

p->dog = new Dog;
cout << p->dog->age; // 可以
```

### 不能访问已释放的内存

```cpp
int *p = new int;
*p; // 可以
delete p;
*p; // 错！
```

```cpp
int *p = (int *)malloc(sizeof(int));
*p; // 可以
free(p);
*p; // 错！
```

```cpp
int *func() {
    int arr[10];
    return arr; // 错！
}

int main() {
    int *p = func();
    p[0];  // 错！arr 已经析构，不能通过空悬指针 / 空悬引用继续访问已经析构的对象
}
```

建议改用更安全的 array 或 vector 容器

```cpp
array<int, 10> func() {
    array<int, 10> arr;
    return arr;
}

int main() {
    auto arr = func();
    arr[0];  // 可以，访问到的是 main 函数局部变量 arr，是对 func 中原 arr 的一份拷贝
}
```

### new / new[] / malloc 和 delete / delete[] / free 必须匹配

```cpp
int *p = new int;
free(p);  // 错！
```

```cpp
int *p = (int *)malloc(sizeof(int));
free(p);  // 正确
```

```cpp
int *p = new int[3];
delete p; // 错！
```

```cpp
int *p = new int[3];
delete[] p; // 正确
```

```cpp
vector<int> a(3);
unique_ptr<int> a = make_unique<int>(42);
```

### 不要访问已经析构的对象

```cpp
struct C {
    int i;
    ~C() { i = 0; }
};

C *c = (C *)malloc(sizeof(C));
cout << c->i; // 可以
c->~C();
cout << c->i; // 错！
free(c);
```

```cpp
std::string func() {
    std::string s = "hello";
    std::string s2 = std::move(s);
    return s;  // 语言：OK，标准库作者：s 不一定是空字符串
}
```

### 不能把函数指针转换为普通类型指针解引用

```cpp
void func() {}

printf("*func = %d\n", *((int *)func));  // 错误
```

> C++ 内存模型是哈佛架构（代码与数据分离），不是冯诺依曼架构（代码也是数据）

## 库函数类

### ctype.h 中一系列函数的字符参数，必须在 0~127 范围内（即只支持 ASCII 字符）

```cpp
isdigit('0');    // 可以，返回 true
isdigit('a');    // 可以，返回 false
isdigit('\xef'); // 错！结果未定义，在 MSVC 的 Debug 模式下会产生断言异常

char s[] = "你好A"; // UTF-8 编码的中文
// "你好a"？
std::transform(std::begin(s), std::end(s), std::begin(s), ::tolower); // 错！结果未定义，因为 UTF-8 编码会产生大于 128 的字节
```

MSVC STL 中 is 系列函数的断言：

`assert(-1 <= c && c < 256);`

理论上可以这样断言：

`assert(0 <= c && c <= 127);`

解决方法：要么改用 iswdigit（MSVC：0-65536，GCC：0-0x010ffff）

```cpp
iswdigit('0');       // 可以，返回 true
iswdigit('\xef');    // 可以，返回 false
iswspace(L'\ufeff'); // 可以，UTF-8 locale 时返回 true，ASCII locale 时返回 false
```

要么自己实现判断

```cpp
if ('0' <= c && c <= '9')  // 代替 isdigit(c)
if (strchr(" \n\t\r", c))  // 代替 isspace(c)
```

### memcpy 函数的 src 和 dst 不能为空指针

```cpp
void *dst = nullptr;
void *src = nullptr;
size_t size = 0;
memcpy(dst, src, size); // 错！即使 size 为 0，src 和 dst 也不能为空指针
```

可以给 size 加个判断

```cpp
void *dst = nullptr;
void *src = nullptr;
size_t size = 0;
if (size != 0) // 可以
    memcpy(dst, src, size);
```

### memcpy 不能接受带有重叠的 src 和 dst

```cpp
char arr[10];
memcpy(arr, arr + 1, 9); // 错！有的同学，以为这个是对的？错了，memcpy 的 src 和 dst
memcpy(arr + 1, arr, 9); // 错！
memcpy(arr + 5, arr, 5); // 可以
memcpy(arr, arr + 5, 5); // 可以
```

如需拷贝带重复区间的内存，可以用 memmove

```cpp
char arr[10];
memmove(arr, arr + 1, 9); // 可以
memmove(arr + 1, arr, 9); // 可以
memmove(arr + 5, arr, 5); // 可以
memmove(arr, arr + 5, 5); // 可以
```

从 memcpy 的 src 和 dst 指针参数是 restrict 修饰的，而 memmove 没有，就可以看出来，memcpy 不允许任何形式的指针重叠，无论先后顺序

### v.back() 当 v 为空时是未定义行为

```cpp
std::vector<int> v = {};
int i = v.back();                  // 错！back() 并不会对 v 是否有最后一个元素做检查，此处相当于解引用了越界的指针
int i = v.empty() ? 0 : v.back();  // 更安全，当 v 为空时返回 0
```

### vector 的 operator[] 当 i 越界时，是未定义行为

```cpp
std::vector<int> v = { 1, 2, 3 };
v[3]; // 错！相当于解引用了越界的指针
```

可以用 at 成员函数

```cpp
std::vector<int> v = { 1, 2, 3 };
v.at(3); // 安全，会检测到越界，抛出 std::out_of_range 异常
```

### 容器迭代器失效

```cpp
std::vector<int> v = { 1, 2, 3 };
auto it = v.begin();
v.push_back(4); // push_back 可能导致扩容，会使之前保存的 v.begin() 迭代器失效
*it = 0;        // 错！
```

如果不需要连续内存，可以改用分段内存的 deque 容器，其可以保证元素不被移动，迭代器不失效。

```cpp
std::deque<int> v = { 1, 2, 3 };
auto it = v.begin();
v.push_back(4); // deque 的 push_back 不会导致迭代器失效
*it = 0;        // 可以
```

- https://www.geeksforgeeks.org/iterator-invalidation-cpp
- https://en.cppreference.com/w/cpp/container

### 容器元素引用失效

```cpp
std::vector<int> v = {1, 2, 3};
int &ref = v[0];
v.push_back(4); // push_back 可能导致扩容，使元素全部移动到了新的一段内存，会使之前保存的 ref 引用失效
ref = 0;        // 错！
```

如果不需要连续内存，可以改用分段内存的 deque 容器，其可以保证元素不被移动，引用不失效。

```cpp
std::deque<int> v = {1, 2, 3};
int &ref = v[0];
v.push_back(4); // deque 的 push_back 不会导致元素移动，使引用失效
ref = 0;        // 可以
```

## 多线程类

### 多个线程同时访问同一个对象，其中至少一个线程的访问为写访问，是未定义行为（俗称数据竞争）

```cpp
std::string s;

void t1() {
    s.push_back('a'); // 写访问，出错！
}

void t2() {
    cout << s.size(); // 读访问
}
```

```cpp
std::string s;

void t1() {
    s.push_back('a'); // 写访问，出错！
}

void t2() {
    s.push_back('b'); // 写访问，出错！
}
```

更准确的说法是：多个线程（无 happens before 关系地）访问同一个对象，其中至少一个线程的访问带有副作用（写访问或带有volatile的读访问），是未定义行为

```cpp
// 八股文教材常见的错误写法！volatile 并不保证原子性和内存序，这样写是有未定义行为的。正确的做法是改用 std::atomic<int>
volatile int ready = 0;
int data;

void t1() {
    data = 42;
    ready = 1;
}

void t2() {
    while (ready == 0)
        ;
    printf("%d\n", data);
}
```

建议利用 mutex，counting_semaphore，atomic 等多线程同步工具，保证多个线程访问同一个对象时，顺序有先有后，不会“同时”发生，那就是安全的

```cpp
std::string s;
std::mutex m;

void t1() {
    std::lock_guard l(m);
    s.push_back('a'); // 有 mutex 保护，可以
}

void t2() {
    std::lock_guard l(m);
    s.push_back('b'); // 有 mutex 保护，可以
}
```

在上面的例子中，互斥锁保证了要么 t1 happens before t2，要么 t2 happens before t1，不会“同时”访问，是安全的

```cpp
std::string s;
std::counting_semaphore<1> sem(1);

void t1() {
    s.push_back('a');
    sem.release(); // 令 t2 必须发生在 t1 之后
}

void t2() {
    sem.acquire(); // t2 必须等待 t1 release 后，才能开始执行
    s.push_back('b');
}
```

在上面的例子中，信号量保证了 t1 happens before t2，不会“同时”访问，是安全的

```cpp
std::string s;
std::atomic<bool> ready{false};

void t1() {
    s.push_back('a');
    ready.store(true, std::memory_order_release); // 令 s 的修改对其他 acquire 了 ready 的线程可见
}

void t2() {
    while (!ready.load(std::memory_order_acquire)) // t2 必须等待 t1 store 后，才能开始执行
        ;
    s.push_back('b');
}
```

在上面的例子中，原子变量的 acquire/release 内存序保证了 t1 happens before t2，不会“同时”访问，是安全的

### 多个线程同时对两个 mutex 上锁，但顺序相反，会产生未定义行为（俗称死锁）

```cpp
std::mutex m1, m2;

void t1() {
    m1.lock();
    m2.lock(); // 错！
    m2.unlock();
    m1.unlock();
}

void t2() {
    m2.lock();
    m1.lock(); // 错！
    m1.unlock();
    m2.unlock();
}
```

解决方法：不要在多个 mutex 上同时上锁，如果确实要多个 mutex，保证顺序一致

```cpp
std::mutex m1, m2;

void t1() {
    m1.lock();
    m2.lock();
    m2.unlock();
    m1.unlock();
}

void t2() {
    m1.lock();
    m2.lock();
    m2.unlock();
    m1.unlock();
}
```

或使用 std::lock

```cpp
std::mutex m1, m2;

void t1() {
    std::lock(m1, m2);
    std::unlock(m1, m2);
}

void t2() {
    std::lock(m2, m1);
    std::unlock(m2, m1);
}
```

### 对于非 recursive_mutex，同一个线程对同一个 mutex 重复上锁，会产生未定义行为（俗称递归死锁）

```cpp
std::mutex m;

void t1() {
    m.lock();
    m.lock();     // 错！
    m.try_lock(); // 错！try_lock 也不允许！
    m.unlock();
    m.unlock();
}

void t2() {
    m.try_lock(); // 可以
}
```

解决方法：改用 recursive_mutex，或使用适当的条件变量

```cpp
std::recursive_mutex m;

void t1() {
    m.lock();
    m.lock(); // 可以
    m.try_lock(); // 可以，返回 true
    m.unlock();
    m.unlock();
    m.unlock();
}
```

# 总结

- 不要玩空指针
- 不要越界，用更安全的 at，subspan 等
- 不要不初始化变量（auto-idiom）
- 开启 `-Werror=return-type`
- 不要重复上锁 mutex
- 仔细看库函数的文档
- 用智能指针管理单个对象
- 用 vector 管理多个对象组成的连续内存
- 避免空悬引用
- 开 Debug 模式的 STL

指定 CMake 的模式：`cmake -B build -DCMAKE_BUILD_TYPE=Debug`

- Debug: `-O0 -g` 编译选项
- Release: `-O3 -DNDEBUG` 编译选项

指定 MSVC 的模式：`cmake --build build --config Debug`

- Debug: 生成 `zenod.dll`，链接 Debug 的 ABI
- Release: 生成 `zeno.dll`，链接 Release 的 ABI

## CppCon 相关视频

顺便推个 CppCon 小视频：https://www.youtube.com/watch?v=ehyHyAIa5so

> {{ icon.tip }} 标题是《CppCon 2017: Piotr Padlewski “Undefined Behaviour is awesome!”》（爆孝）
