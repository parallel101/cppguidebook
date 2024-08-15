# 重新认识声明与定义（未完工）

[TOC]

## 我们要牢记白指导说的道理

> {{ icon.fun }} mq 白在[川上](https://github.com/parallel101/cppguidebook/pull/23)曰：
>
> > 非定义声明，因为 Game 在此处为不完整类型
>
> 我能明白其意思，定义一定是声明，声明却不一定是定义。所以用了：“非定义声明”这个词语，很专业的措辞。
>
> 不过我觉得大多数普通开发者并不够清楚这一点，看到这段注释同样会感到疑惑。
>
> 在他们眼里声明和定义是两种东西，此处如果直接用声明它们可能就不会有理解问题了。例如：“只是声明，不是定义”之类的措辞。
>
> 或许我们应该考虑在保证专业以及严谨的情况下，稍微补充解释一下“非定义声明”这个用词。

## 多文件编译的必要性

## 翻译单元 (TU)

## 符号的链接类型 (linkage)

函数和变量，在对外的可见性这方面，有以下几种类型：

- 外部链接 (ODR external linkage)：对其他翻译单元可见
- 共享链接 (non-ODR external linkage)
- 内部链接 (internal linkage)
- 无链接 (no linkage)

函数和变量的可见性这一属性，被 C++ 官方称为链接（linkage），是因为符号的可见性处理通常是链接器（ld）负责的，不同类型链接（linkage）的效果，在链接（link）的时候才会生效。

定义在全局（名字空间）中的情况：

```cpp
int i;                  // 变量声明并定义为“外部链接”
int f(int x);           // 函数声明为“外部链接”
int f(int x) {}         // 函数声明并定义为“外部链接”

extern int i;           // 变量声明为“外部链接”
extern int f(int x);    // 函数声明为“外部链接”
extern int f(int x) {}  // 函数声明并定义为“外部链接”

inline int i;           // 变量声明并定义为“共享链接”
inline int f(int x);    // 函数声明为“共享链接”
inline int f(int x) {}  // 函数声明并定义为“共享链接”

static int i;           // 变量声明并定义为“内部链接”
static int f(int x);    // 函数声明为“内部链接”
static int f(int x) {}  // 函数声明并定义为“内部链接”
```

定义在类（class）中的情况：

```cpp
struct Class {

int i;                  // 变量声明并定义为“无链接”
int f(int x);           // 函数声明为“外部链接”
int f(int x) {}         // 函数声明并定义为“共享链接”

inline static int i;           // 变量声明并定义为“共享链接”
inline static int f(int x);    // 函数声明为“共享链接”
inline static int f(int x) {}  // 函数声明并定义为“共享链接”

static int i;           // 变量声明并定义为“外部链接”
static int f(int x);    // 函数声明为“外部链接”
static int f(int x) {}  // 函数声明并定义为“外部链接”

};
```
