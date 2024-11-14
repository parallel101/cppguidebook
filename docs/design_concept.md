# 鸭子类型与 C++20 concept (未完工)

[TOC]

如果一个东西叫起来像一只鸭，走起路来像一只鸭，那么不妨认为他就是一只鸭。

## 为什么需要多态

我们有三种类型的狗：拉布拉多犬，藏獒，张心欣。

> {{ icon.fun }} 请勿侮辱拉布拉多和藏獒！

他们有一个共同点，那就是它们都会狗叫（bark）以及自我介绍（intro）。

```cpp
struct Labrador {
    void intro() {
        puts("我能帮你捡回棍棍！");
    }

    void bark() {
        puts("汪汪！");
    }
};

struct Mastiff {
    void intro() {
        puts("我能保卫国王荣耀！");
    }

    void bark() {
        puts("汪汪！");
    }
};

struct Xinxin {
    void intro() {
        puts("我能祝您裁员滚滚！");
    }

    void bark() {
        puts("从未贡献任何核心功能！");
    }
};
```

现在，我们需要设计一个“饲养员”函数，他会让狗狗先自我介绍，然后叫两声。

传统的基于重载的写法，需要连续写三个一模一样的函数体，非常麻烦，违反“避免重复”原则，不利于代码未来的维护。

```cpp
void feeder(Labrador dog) {
    dog.intro();
    dog.bark();
    dog.bark();
}

void feeder(Mastiff dog) {
    dog.intro();
    dog.bark();
    dog.bark();
}

void feeder(Xinxin dog) {
    dog.intro();
    dog.bark();
    dog.bark();
}
```

这种写法的缺陷在两个方面：

1. 当需要添加一个新类型的狗狗 `Shiba` 时，需要再复制粘贴定义一个 `feeder(Shiba)` 的重载。
2. 当需要修改 `feeder` 的操作内容时，需要把三个重载都同样地修改一遍。

### 模板函数

可以把 `feeder` 定义为模板函数，这样他的参数可以为任意类型。

只要传入的模板参数类型具有 `intro` 和 `bark` 这两个成员函数，编译就不会出错。

```cpp
template <typename Dog>
void feeder(Dog dog) {
    dog.intro();
    dog.bark();
    dog.bark();
}
```

1. 当添加了一个新类型的狗狗 `Shiba` 时，什么都不用做，只要 `Shiba` 定义了 `intro` 和 `bark` 成员函数，就可以直接传入 `feeder`，无需做任何适配工作。
2. 当我们需要修改 `feeder` 的操作内容时，只需修改这一个模板函数的内容就行了。

可见，模板函数是重载函数的便民版，可用于当所有的重载函数内部代码完全一致的情况。

### “特殊照顾”

但是，如果有些特殊类型的重载需要特殊照顾，导致内部代码不一样，这种传统的模板函数就不适用了。

例如，我们现在让拉布拉多和藏獒保持原样，但张心欣因为智力原因，没有了 `intro` 的功能。

```cpp
struct Labrador {
    void intro() {
        puts("我能帮你捡回棍棍！");
    }

    void bark() {
        puts("汪汪！");
    }
};

struct Mastiff {
    void intro() {
        puts("我能保卫国王荣耀！");
    }

    void bark() {
        puts("汪汪！");
    }
};

struct Xinxin {
    // 没有 intro() 成员函数

    void bark() {
        puts("从未贡献任何核心功能！");
    }
};
```

对于张心欣这种智力特殊丧失 `intro` 功能的狗狗，需要特殊照顾，在 `feeder` 中需要对 `Xinxin` 做特别判断，如果判断到狗狗类型是 `Xinxin`，就需要跳过对 `intro` 的调用，这该怎么做呢？

传统的暴力重载函数的方法中，很简单，只需要拉布拉多和藏獒的重载版本保持不变，只对张心欣这一个 `feeder(Xinxin)` 重载里的代码做特殊修改，删掉 `intro` 调用即可。

```cpp
void feeder(Labrador dog) {
    dog.intro();
    dog.bark();
    dog.bark();
}

void feeder(Mastiff dog) {
    dog.intro();
    dog.bark();
    dog.bark();
}

void feeder(Xinxin dog) {
    // dog.intro();
    dog.bark();
    dog.bark();
}
```

模板函数要如何实现这种“特殊照顾”呢？

#### 模板函数与普通重载函数并列

一种方法是额外定义一个普通的重载函数 `feeder(Xinxin)`，与模板函数并列。

```cpp
template <typename Dog>
void feeder(Dog dog) {
    dog.intro();
    dog.bark();
    dog.bark();
}

void feeder(Xinxin dog) {
    // dog.intro();
    dog.bark();
    dog.bark();
}
```

当调用 `feeder` 时，得益于 C++ 的重载机制，会优先匹配非模板的普通重载函数，如果匹配不成功，才会落入通用的模板函数作为备选方案。

```cpp
Xinxin xinxin;
feeder(xinxin);   // 会优先匹配到 feeder(Xinxin) 这个普通函数
Labrador labrador;
feeder(labrador); // 会匹配到 feeder<Labrador>(Labrador) 这个模板函数
```

该方案依然存在缺陷：

1. 这里模板函数和普通函数中，最后都有两次 `bark` 调用，出现了代码重复。
2. 如果我们想要添加一个新类型的狗狗 `Yuanming`，他也没有 `intro`，难道又要为他单独定义一个重载么？

#### 模板函数内做 if 特殊判断

针对缺点 1，我们想到，能不能不用分离两个函数，而是在函数内部，动态判断模板参数 `Dog` 类型是否为 `Xinxin`，如果不是 `Xinxin` 才去调用 `dog.intro()`。

```cpp
template <typename Dog>
void feeder(Dog dog) {
    if (Dog != Xinxin) {
        dog.intro();
    }
    dog.bark();
    dog.bark();
}
```

上面这种写法 `Dog != Xinxin` 仅为示意，实际上是编译不通过的。

因为只有值表达式才能用运算符 `!=` 比较，类型表达式不能用 `!=` 比较。

要两个类型是否相等，需要用到 `<type_traits>` 头文件中的 `is_same_v`。

`is_same_v<X, Y>` 相当于类型版本的 `X == Y`。

这里因为我们要判断的是不等，`Dog != Xinxin`，所以用 `!is_same_v<Dog, Xinxin>` 即可。

```cpp
#include <type_traits>

template <typename Dog>
void feeder(Dog dog) {
    if (!std::is_same_v<Dog, Xinxin>) {
        dog.intro();
    }
    dog.bark();
    dog.bark();
}
```

试试看，你会发现编译错误：

```cpp
Labrador labrador;
feeder(labrador); // 编译通过
Xinxin xinxin;
feeder(xinxin);   // 编译报错：“Xinxin 没有成员 intro”
```

为什么？我们不是判断了 `if (Dog 不为 Xinxin)` 才会调用 `dog.intro()` 吗？我们现在传入的是一个 `Xinxin` 类型的狗狗，为什么还是会执行到 `dog.intro()` 导致编译器找不到这个成员函数而报错呢？

原来，“执行到”和“编译到”是两个概念。

`if` 只是避免了运行时的“执行到”，但编译期还是会“编译到”的。

例如以下代码会出错：

```cpp
if (0) {
    "string" = 0;
}
```

虽然 `if` 的判断条件始终为 `false`，“运行时”永远不会执行到里面的代码，但是由于编译器编译时，每个他看到的代码都要生成相应的 IR 中间码，即使最终可能被优化掉，也要为其生成 IR。

所以虽然 `if (0)` 会让运行时永远无法执行到或者可能被“中后端”优化掉而不会产生汇编码，但编译器的“前端”仍需完成该分支体内代码的翻译工作，而 `"string" = 0` 是非法的，根本无法生成出 IR 中间码，导致编译出错终止。

### C++17 编译期分支 `if constexpr`

为了避免在“编译期”就触及 `xinxin.intro()` 这个无法通过编译的代码，我们需要使在编译期就完成分支，而不是拖到运行时或优化时。

C++17 引入的 `if constexpr` 就是一个编译期版本的 `if` 分支，他要求判断的表达式必须是编译期可以确定的，并且能保证分支一定在编译期完成，保证不会在运行时生成的汇编中产生任何额外的分支指令，无论是否开启优化。

如果 `if constexpr` 的分支条件不满足，则分支内的代码根本不会进行编译，即使含有本不能通过编译的代码也不会报错了。

```cpp
#include <type_traits>

template <typename Dog>
void feeder(Dog dog) {
    if constexpr (!std::is_same_v<Dog, Xinxin>) { // 编译期决定要不要编译下面的代码
        dog.intro();
    }
    dog.bark();
    dog.bark();
}
```

如果 Dog 是 Xinxin，则 `dog.intro()` 这条语句从编译期前端开始就不会经过编译，无论是否开启优化都会被抹除。因此即使找不到 `intro` 这个成员，也绝对不会报错了。

> {{ icon.fun }} 抹除的就和张心欣抹除小彭老师贡献一样干净！

#### 依然无法自动适配所有新增类型

缺点 2 依然存在：如果我们想要添加一个新类型的狗狗 `Yuanming`，他也没有 `intro`，那就得在 if 判断中添加一个 `is_same_v<Dog, Yuanming>` 判断，每多一个没有 `intro` 的狗狗就得加一遍，没完没了。

```cpp
struct Xinxin {
    // 没有 intro() 成员函数

    void bark() {
        puts("从未贡献任何核心功能！");
    }
};

struct Yuanming {
    // 没有 intro() 成员函数

    void bark() {
        puts("Taichi is your hobby, but yuanming's work");
    }
};

template <typename Dog>
void feeder(Dog dog) {
    if constexpr (!std::is_same_v<Dog, Xinxin>
               && !std::is_same_v<Dog, Yuanming>) { // 搁着叠罗汉呢？
        dog.intro();
    }
    dog.bark();
    dog.bark();
}
```

## C++20 concepts

### `requires` 检查表达式合法性

与其用 `is_same_v` 一个个罗列出“没有 `intro`”的类型一一判断，不如直接检测 `dog` 有没有 `intro` 这个成员。

C++20 引入的 `requires` 关键字，可以帮你检测一个表达式是否“合法”，也就是能不能编译通过，如果能编译通过，会返回 `true`。

过去，如果一个表达式非法（例如找不到成员函数），我们就只能眼巴巴让编译器出错终止编译……

```cpp
struct Xinxin {
    void bark() {
        puts("从未贡献任何核心功能！");
    }
};

Xinxin xinxin;
xinxin.intro(); // 编译出错
xinxin.bark();  // 编译通过
```

现在，我们可以把“编译是否通过”安全地作为一个 `bool` 值返回回来，供我们后续判断处理，而不必粗暴地终止整个编译。

> {{ icon.fun }} 过去：一坨史害了一锅粥。现在：每粒米都放在一个隔离的“安全沙盒”里独立检验，检验结果通过 `bool` 返回，告诉小彭老师要不要吃这粒米。

用法就是：`requires { 要检验的表达式; }`

所以，我们可以用 `requires { 要检验的变量.要检验的成员函数(参数...); }` 来判断某个变量类型是否有特定名字的成员函数，因为如果没有，那么表达式编译会失败，`requires` 就会返回 `false`。所以只要这个 `requires` 返回了 `true`，就可以说明该类型含有此名称的成员变量或成员函数了。对于成员函数还需要注意指定正确类型的参数，否则也无法通过编译。利用此方法还可以检测成员函数是否支持特定参数类型的重载等。

```cpp
struct Xinxin {
    void bark() {
        puts("从未贡献任何核心功能！");
    }
};

Xinxin xinxin;
bool has_intro = requires { xinxin.intro(); }; // false
bool has_bark = requires { xinxin.bark(); };   // true
```

`requires` 判断的结果是编译期常量（`constexpr bool`），可以作为 `if constexpr` 的条件使用。

结合 `if constexpr` 可以根据一个类型有没有某个成员（通过检测访问这个成员是否可以编译通过）来决定要不要调用这个成员。

```cpp
if constexpr (requires { dog.intro(); }) {
    dog.intro();
}
```

```cpp
template <typename Dog>
void feeder(Dog dog) {
    if constexpr (requires { dog.intro(); }) { // 如果支持 .intro() 成员函数
        dog.intro(); // 则调用他
    }
    dog.bark();
    dog.bark();
}
```

`if constexpr` 还可以带有 `else`，甚至 `else if constexpr`。

```cpp
template <typename Dog>
void feeder(Dog dog) {
    if constexpr (requires { dog.intro(); }) { // 如果支持 .intro() 成员函数
        dog.intro(); // 则调用他
    } else if constexpr (requires { dog.intro(1); }) { // 如果支持 .intro(int) 这种带一个 int 参数的重载
        dog.intro(1); // 则尝试调用这种带有 int 参数的重载
    } else {
        puts("此狗狗似乎不支持自我介绍呢"); // 否则打印警告信息
    }
    dog.bark();
    dog.bark();
}
```

### `requires` 应用案例：迭代器

众所周知，迭代器分为很多类型，例如：

随机迭代器支持 `+=` 操作，可以向前步进任意整数格，也可以 `-=` 向后退步。

而前向迭代器只能 `++` 向前移动一格，如果需要向前移动 n 格，就需要重复执行 `++` n 次。

还有一种双向迭代器，他既可以 `++` 向前移动一格，也可以 `--` 向后退步一格，但是不支持任意整数步长的 `+=` 和 `-=`，需要用循环来模拟。

比如 `vector` 的迭代器就属于随机迭代器，因为 `vector` 是连续内存的容器，他是一个线性的数组，其迭代器实际上就是一个指向元素的指针，迭代器的步进实际上就是指针在 `+=`，当然支持前进（加上）任意整数 n 格了。

我们现在想要实现一个通用的迭代器“步进”函数 `advance`：

1. 对于随机迭代器他会直接调用 `+=` 前进 n 步，不用循环一格格 `++` 的低效。
2. 对于前向迭代器他会循环调用 `++` n 次，如果 n 为负数则报错。
3. 对于双向迭代器他会循环调用 `++` 或 `--` n 次，取决于 n 是否为正数。

伪代码如下：

```cpp
template <typename It>
void advance(It &it, int n) {
    if (随机迭代器) {
        it += n;

    } else if (双向迭代器) {
        if (n > 0) {
            for (int i = 0; i < -n; ++i) {
                --it;
            }
        } else {
            for (int i = 0; i < n; ++i) {
                ++it;
            }
        }

    } else { // 前向迭代器
        if (n < 0) throw "前向迭代器不能步进一个负数";
        for (int i = 0; i < n; ++i) {
            ++it;
        }
    }
}
```

如何用 `requires` 和 `if constexpr` 实现这个效果？

```cpp
template <typename It>
void advance(It &it, int n) {
    if constexpr (requires { it += n; }) {
        it += n;

    } else if constexpr (requires { ++it; --it; }) {
        if (n > 0) {
            for (int i = 0; i < -n; ++i) {
                --it;
            }
        } else {
            for (int i = 0; i < n; ++i) {
                ++it;
            }
        }

    } else {
        if (n < 0) throw "前向迭代器不能步进一个负数";
        for (int i = 0; i < n; ++i) {
            ++it;
        }
    }
}
```

是不是很简单呢？只需要注意到随机迭代器需要支持 `+=`，那么我们就通过 `requires` 判断 `it` 是否支持 `+=`，支持了就说明应该是一个随机迭代器。否则如果是双向迭代器就应该支持 `++` 和 `--`，那就采用双向迭代器的方案。否则就只可能是前向迭代器，当 `n < 0` 时需要报错因为他不支持 `++`。

这里我们用了 `requires { ++it; --it; }` 这种带有多条语句的写法。没错，`requires` 支持一次性判断多条语句是否合法，只要其中一条非法就会返回 `false`，必须全部满足了才能返回 `true`。

> {{ icon.tip }} 所以 `requires { ++it; --it; }` 实际上等价于 `requires { ++it; } && requires { --it; }`

### `requires` 自带干粮

有时，我们的 `requires` 是在一个函数体内，已经有变量 `dog` 的情况下，这时只需写 `requires { dog.intro(); }` 即可判断 `dog` 是否支持 `dog.intro()` 成员函数。

```cpp
Dog dog;
bool has_intro = requires { dog.intro(); };
```

但有时，我们需要直接判断 `Dog` 类型是否含有成员函数 `intro`，避免在函数中创建 `Dog dog` 变量。

一种粗暴的方法是，直接用 `Dog()` 就地构造出一个 `Dog` 类型的对象来，然后访问这个临时对象的 `intro()`。

```cpp
bool has_intro = requires { Dog().intro(); };
```

但是，这会要求 `Dog` 支持默认构造函数 `Dog()`，如果 `Dog` 不支持默认构造，比如需要两个参数 `Dog(1, 2)` 这样才能构造出来，那么 `Dog().intro()` 编译就会出错，即使 `Dog` 有 `.intro()` 成员函数也会出错，因为前面的 `Dog()` 就编译不过，导致明明有 `.intro()` 却返回了 `false`。

所以 `requires` 提供了一种方便的语法糖，你可以在 `requires` 和 `{` 之间加入 `(...)`，其中用类似于函数参数定义的写法，写你需要用到的变量的定义，在 `{...}` 中可以使用这些变量，变量的类型就是你在 `(...)` 中定义的类型。

```cpp
bool has_intro = requires (Dog dog) { dog.intro(); };
```

和需要在函数体内定义一个 `Dog dog` 变量再判断相比，`requires (Dog dog)` 这种写法仅仅只是构造出一个“编译期”象征性创建的“虚假”变量，仅供判断使用，并不会在栈上产生任何实际的空间占用，不增加任何运行时成本。

`()` 中也可以有多个变量的定义，用逗号分隔：

```cpp
if constexpr (requires (It it, int n) {
    it += n;
}) {
    // 检测到随机迭代器时要执行的分支
}
```

### 预定义好 `concept` 更方便

推荐把常用到的条件预先定义成 `constexpr bool` 变量模板，这样以后不用每次都重写所有需要判断的表达式了。

```cpp
template <typename It>
conetexpr bool random_access_iterator = requires (It it, int n) {
    it += n;
    it -= n;
    ++it;
    --it;
};
```

`conetexpr bool` 表示这是一个编译期就能确定值的变量，不会占用任何运行时空间。只要 `It` 这个模板参数确定，`bool` 的值就是编译期唯一确定的。

更好的写法是用 `concept` 作为 `constexpr bool` 的简写，看起来更加“专业”“高B格”。

> {{ icon.tip }} `concept` 并不只是个缩写，他还附赠了一些额外的好处，稍后介绍。但总的来说 `concept` 完全可以当作普通 `bool` 使用。

预先定义好 `concept`，以后使用就不用“烧脑”思考需要支持哪些成员函数了，直接报 `concept` 的名字就行。

这也是“概念 (concept)”得名的由来：如果一个类型支持“鸭叫”，那么他就符合“鸭子”这个概念。

```cpp
template <typename It>
concept random_access_iterator = requires (It it, int n) {
    it += n;
    it -= n;
    ++it;
    --it;
};

template <typename It>
concept bidirectional_iterator = requires (It it) {
    ++it;
    --it;
};

template <typename It>
concept forward_iterator = requires (It it) {
    ++it;
};

template <typename It>
void advance(It &it, int n) {
    if constexpr (random_access_iterator<It>) {
        it += n;

    } else if constexpr (bidirectional_iterator<It>) {
        if (n > 0) {
            for (int i = 0; i < -n; ++i) {
                --it;
            }
        } else {
            for (int i = 0; i < n; ++i) {
                ++it;
            }
        }

    } else if constexpr (forward_iterator<It>) {
        if (n < 0) throw "前向迭代器不能步进一个负数";
        for (int i = 0; i < n; ++i) {
            ++it;
        }

    } else {
        throw "It 不是任何一种合法的迭代器";
    }
}
```

针对满足不同的概念的参数类型采取不同的操作方式，这就是 C++20 概念。

### 标准库定义好的 `<concepts>`

我们自己定义的概念，难免有一些疏漏。例如前向迭代器实际上不仅要求支持 `++it` 还要支持 `it++` 这种后置的自增运算符，还要求支持拷贝构造函数等。

所以，对于这种标准库就已经有的概念，推荐使用标准库头文件 `<concepts>` 中定义好的概念来用，而不必一个个自己手动定义。

```cpp
#include <concepts> // 定义了 concept 如 std::random_access_iterator

if constexpr (std::random_access_iterator<It>) {
    ... // 如果满足随机迭代器概念
}
```

一些 C++11 `<type_traits>` 中就有的一些判断用的 `constexpr bool` 变量模板，在 C++20 `<concepts>` 中也“转正”为了 `concept`：

```cpp
#include <type_traits>

if constexpr (std::is_integral_v<T>) {  // C++17
    // T 是整数类型时
}

#include <concepts>

if constexpr (std::integral<T>) {       // C++20
    // T 是整数类型时
}
```

其实际效果是相同的，只是名字更简洁，并且类型由 `constexpr bool` 变成了 `concept`。

> {{ icon.tip }} 憋担心，如果你的编译器不支持 C++20，用 `std::is_integral_v` 实际上也和 `std::integral` 概念是一样的。

> {{ icon.fun }} `<concepts>` 基本上就是去掉了 `is_` 和 `_v`。

这里我们罗列一部分常见的 `concept` 和老 `constexpr bool` 对应列表：

| `<concepts>` | `<type_traits>` |
|--------------|-----------------|
| `std::same_as<T, U>` | `std::is_same_v<T, U>` |
| `std::derived_from<Base, Derived>` | `std::is_base_of_v<Base, Derived>` |
| `std::convertible_to<From, To>` | `std::is_convertible_v<From, To>` |
| `std::integral<T>` | `std::is_integral_v<T>` |
| `std::floating_point<T>` | `std::is_floating_point_v<T>` |
| `std::signed_integral<T>` | `std::is_signed_v<T>` |
| `std::unsigned_integral<T>` | `std::is_unsigned_v<T>` |
| `std::move_constructible<T>` | `std::is_move_constructible_v<T>` |
| `std::copy_constructible<T>` | `std::is_copy_constructible_v<T>` |
| `std::copy_assignable<T>` | `std::is_copy_assignable_v<T>` |
| `std::move_assignable<T>` | `std::is_move_assignable_v<T>` |
| `std::copyable<T>` | `std::is_copy_constructible_v<T> && std::is_copy_assignable_v<T>` |
| `std::movable<T>` | `std::is_move_constructible_v<T> && std::is_move_assignable_v<T>` |
| `std::constructible_from<T, Args...>` | `std::is_constructible_v<T, Args...>` |
| `std::assignable_from<T, U>` | `std::is_assignable_v<T, U>` |
| `std::default_initializable<T>` | `std::is_default_constructible_v<T>` |
| `std::destructible<T>` | `std::is_destructible_v<T>` |
| `std::semiregular<T>` | `std::is_default_constructible_v<T> && std::is_copy_constructible_v<T> && std::is_copy_assignable_v<T>` |
| `std::regular<T>` | `std::semiregular<T> && std::is_equality_comparable_v<T>` |
| `std::equality_comparable<T>` | `std::is_equality_comparable<T>` |
| `std::totally_ordered<T>` | `std::equality_comparable<T> && std::is_less_comparable<T>` |
| `std::input_or_output_iterator<It>` | `std::is_base_of_v<std::input_iterator_tag, typename std::iterator_traits<It>::iterator_category> \|\| std::is_base_of_v<std::output_iterator_tag, typename std::iterator_traits<It>::iterator_category>` |
| `std::input_iterator<It>` | `std::is_base_of_v<std::input_iterator_tag, typename std::iterator_traits<It>::iterator_category>` |
| `std::output_iterator<It>` | `std::is_base_of_v<std::output_iterator_tag, typename std::iterator_traits<It>::iterator_category>` |
| `std::forward_iterator<It>` | `std::is_base_of_v<std::forward_iterator_tag, typename std::iterator_traits<It>::iterator_category>` |
| `std::bidirectional_iterator<It>` | `std::is_base_of_v<std::bidirectional_iterator_tag, typename std::iterator_traits<It>::iterator_category>` |
| `std::random_access_iterator<It>` | `std::is_base_of_v<std::random_access_iterator_tag, typename std::iterator_traits<It>::iterator_category>` |
| `std::invocable<F, Args...>` | `std::is_invocable_v<F, Args...>` |

更多细节请自行前往 [cppreference](https://en.cppreference.com/w/cpp/concepts) 慢慢查阅。
