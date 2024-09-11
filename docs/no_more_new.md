# 现代 C++ 从拒绝 new 开始

使用 new 和 delete 是一种过时的内存管理方式，容易导致内存泄漏和悬空指针，应当永不使用。

优秀的现代 C++ 项目，都使用**智能指针**和**容器**管理内存，从来不需要直接创建原始指针。

下列三种情况下，你可以使用 new 和 delete：

1. 你在封装一个非常底层的内存分配器库。
2. 你是 C++98 用户，且你的老板不允许使用 boost（其提供了智能指针）。
3. 你想要创造一些内存泄漏来惩罚拖欠工资的脑板。

> 同理，malloc 和 free 也是不允许的。

不仅 new 不应该出现，原始指针也应该少出现，而是更安全，用法更单一的 **引用** 或 **span** 代替。

> {{ icon.tip }} 用法更单一为什么是好事？请看[强类型 API 设计专题章](type_rich_api.md)。

## 案例 1

同学：我想要**分配一段内存空间**，你不让我 new，我还能怎么办呢？

```cpp
char *mem = new char[1024];   // 同学想要 1024 字节的缓冲区
read(1, mem, 1024);           // 用于供 C 语言的读文件函数使用
delete[] mem;                 // 需要手动 delete
```

> {{ icon.fun }} 可以看到，他所谓的“内存空间”实际上就是一个“char 数组”。

小彭老师：有没有一种可能，vector 就可以分配内存空间。

```cpp
vector<char> mem(1024);
read(1, mem.data(), mem.size());
```

vector 一样符合 RAII 思想，构造时自动申请内存，离开作用域时自动释放。

只需在调用 C 语言接口时，取出原始指针：

- 用 data() 即可获取出首个 char 元素的指针，用于传递给 C 语言函数使用。
- 用 size() 取出数组的长度，即是内存空间的字节数，因为我们的元素类型是 char，char 刚好就是 1 字节的，size() 刚好就是字节的数量。

此处 read 函数读完后，数据就直接进入了 vector 中，根本不需要什么 new。

> {{ icon.detail }} 更现代的 C++ 思想家还会用 `vector<std::byte>`，明确区分这是“字节”不是“字符”。如果你读出来的目的是当作字符串，可以用 `std::string`。

> {{ icon.warn }} 注意：一些愚蠢的教材中，用 `shared_ptr` 和 `unique_ptr` 来管理数组，这是错误的。
>
> `shared_ptr` 和 `unique_ptr` 智能指针主要是用于管理“单个对象”的，不是管理“数组”的。
>
> `vector` 一直都是数组的管理方式，且从 C++98 就有。不要看到 “new 的替代品” 只想到智能指针啊！“new [] 的替代品” 是 `vector` 啊！

此处放出一个利用 `std::wstring` 分配 `wchar_t *` 内存的案例：

```cpp
std::wstring utf8_to_wstring(std::string const &s) {
    int len = MultiByteToWideChar(CP_UTF8, 0,
                                  s.data(), s.size(),
                                  nullptr, 0);  // 先确定长度
    std::wstring ws(len, 0);
    MultiByteToWideChar(CP_UTF8, 0,
                        s.data(), s.size(), 
                        ws.data(), ws.size());  // 再读出数据
    return ws;
}
```

### 贴士 1.1

我一般会用更直观的 auto 写法，这样更能明确这是在创建一个 vector 对象，然后保存到 mem 这个变量名中。

```cpp
auto mem = vector<char>(1024);
read(1, mem.data(), mem.size());
```

> {{ icon.tip }} 这被称为 auto-idiom：始终使用 auto 初始化变量，永远别使用可能引发歧义的类型前置。

### 贴士 1.2

有的同学会想当然地提出，用智能指针代替 new。

```cpp
auto mem = make_shared<char[]>(1024);
read(1, mem.get(), 1024);
```

可 new 的替代品从来不只有智能指针一个，也可以是 vector 容器！

- 智能指针只会用于**单个对象**！
- **动态长度的数组**，正常人都是用 vector 管理的。

> 很多劣质的所谓 “现代 C++ 教材” 都忽略了这一点，总是刻意夸大了智能指针的覆盖范围，为了新而新，而对实际上更适合管理 **动态长度内存空间** 的 vector 只字不提。

vector 管理动态长度内存空间的优势：

- 你可以随时随地 resize 和 push_back，加入新元素，而智能指针管理的数组要重新调整大小就比较困难。
- vector 的拷贝构造函数是深拷贝，符合 C++ 容器的一般约定。而 unique_ptr 完全不支持拷贝，深拷贝需要额外的代码，shared_ptr 则是浅拷贝，有时会导致数据覆盖。

其实 `shared_ptr<char[]>` 也不是不可以用，然而，智能指针管理的数组，并不能方便地通过 `.size()` 获取数组的长度，必须用另一个变量单独存储这个长度。这就违背了封装原则，那会使你的代码变得不可维护。

> 绝大多数情况下，可维护性总是比性能重要的，你只需要比较 **你重构代码花的时间** 和 **计算机运行这段代码所需时间** 就明白值不值了。

### 贴士 1.3

如果是其他类型，可能需要乘以 `sizeof(元素类型)`，取决于那个 C 函数要求的是“字节数”还是“元素数”。

```cpp
auto mem = vector<int>(1024);
read(1, mem.data(), mem.size() * sizeof(mem[0]));
auto max = find_max(mem.data(), mem.size());
```

### 贴士 1.4

对于你自己的 C++ 函数，就没必要再提供

TODO: span, gsl::span, boost::span

## 案例 2

同学：我需要在“堆”上分配一个对象，让他持久存在。你不让我用 new，我只能在“栈”上创建临时对象了，如果要返回或存起来的话根本用不了啊。

```cpp
Foo *hello() {
    Foo *foo = new Foo();
    return foo;
}
```

小彭老师：你可以使用智能指针，最适合新人上手的智能指针是 shared_ptr。
当没有任何函数或对象持有该 shared_ptr 指向的对象时，也就是当调用者存储 hello() 返回值的函数体退出时，指向的对象会被自动释放。

```cpp
shared_ptr<Foo> hello() {
    shared_ptr<Foo> foo = make_shared<Foo>();
    return foo;
}
```

总之，这样替换你的代码：

- `T *` 换成 `shared_ptr<T>`
- `new T(...)` 换成 `make_shared<T>(...)`

你的代码就基本上安全了，再也不用手动 delete 了。

> 有个用了 shared_ptr 还会内存泄漏的边缘情况：循环引用，通常是实现双向链表时，weak_ptr 可以解决，稍后介绍。

### 贴士 2.1

unique_ptr 和 shared_ptr 有什么区别？初学者应该先学哪个？

unique_ptr 是独占所有权，他的限制更多，比如：

- 不允许拷贝，只允许移动。
- 不允许赋值，只允许移动赋值。
- 用 unique_ptr 主要是出于性能优势。

> 然而性能总是不如安全重要的，你是想要**一个造在火星的豪华宫殿，还是一个地球的安全老家？**

所以，建议你先全部替换成泛用性强、易用的 shared_ptr。等确实出现性能瓶颈时，再对瓶颈部分单独调试优化也不迟。

> 先把老家造好了，然后再想办法移民火星，而不是反过来。

### 贴士 2.2

有些老式的所谓 “现代 C++ 教程” 中，会看到这样 new 与智能指针并用的写法：

```cpp
shared_ptr<Foo> foo(new Foo());
```

从 C++14 开始，这已经是**过时的**！具有安全隐患（如果构造函数可能抛出异常），且写起来也不够直观。

现在人们一般都会用 make_shared 函数，其内部封装不仅保证了异常安全，而且会使 shared_ptr 的控制块与 Foo 对象前后紧挨着，只需一次内存分配，不仅更直观，还提升了性能。

```cpp
auto foo = make_shared<Foo>();
```

> {{ icon.detail }} 有趣的是，make_shared 在 C++11 就引入了，make_unique 却直到 C++14 才引入。

从 C++14 开始，内存安全的现代 C++ 程序中就不会出现任何显式的 new 了，哪怕是包在 shared_ptr 或 unique_ptr 内的也不行。（除了最上面说的 3 种特殊情况）

### 贴士 2.3

如果你需要调用的 C 语言接口还需要原始指针的话，用 `.get()` 可以从智能指针中获取原始指针。建议只在和 C 语言打交道时 `.get()`，其余时间一律 shared_ptr 保证安全。

```cpp
extern "C" void some_c_function(Foo *foo);

auto foo = make_shared<Foo>();
some_c_function(foo.get());
```


## RAII 比起手动 delete 的优势

在日常代码中，我们常常会使用“如果错误了就提前返回”的写法。这被称为**提前返回 (early-return)**，一种优质的代码写法，比弄个很大的 else 分支要可维护得多。

> {{ icon.detail }} 在[错误处理专题](error_code.md)中有进一步的详解。

然而这有时我们会忘记在提前返回的分支中 delete 之前分配过的所有指向动态内存的指针。

```cpp
int func() {
    Foo *foo = new Foo();
    ...
    if (出错) {
        // 提前返回的分支中忘记 delete foo！
        return -1;
    }
    ...
    delete foo;
    return 0;
}
```

过去，为了解决每个提前返回分支都需要 `delete` 之前分配所有资源的问题，人们使用 `goto` 大法拙劣地避免提前退出时忘记释放内存：

```cpp
int func() {
    Foo *foo1 = nullptr;
    Foo *foo2 = nullptr;
    foo1 = new Foo();
    ...
    foo2 = new Foo();
    ...
    if (出错) {
        goto EXIT;
    }
    ...
EXIT:
    if (nullptr != foo1) {
        delete foo1;
    }
    if (nullptr != foo2) {
        delete foo2;
    }
    return 0;
}
```

这就会让编写程序的人精神紧绷，更加仔细的检查 `new` 和 `delete` 是否配对，`goto EXIT` 会不会有其他负面影响。享受不到编程的快乐就算了，反而在遭罪。。。

此外，`new` 和 `delete` 使用不配套时，也会导致严重的错误。由于 `new` 和 `delete` 是基于指针操作的，但是在很庞大的代码系统中我们拿到一个指针 `data *ptr` 后。这个 `ptr` 是空指针吗？还是已经被释放了吗？还是已经有内存了？还是交由我们申请内存？很容易忘记 `new` 而直接 `delete`，或者对同一个指针 `new` 了两次，或者没有 `delete` 等等。看到屎山代码后的坏心情会导致代码维护难度指数级增加。

而智能指针，不论是提前返回还是最终的返回，只要是函数结束了，都能自动释放。智能指针使得程序员写出“提前返回式”毫无精神压力，再也不用惦记着哪些需要释放。

```cpp
int func() {
    shared_ptr<Foo> foo = make_shared<Foo>();
    ...
    if (出错) {
        return -1;
    }
    ...
    return 0;
}
```

## shared_ptr 小课堂

### 自动释放

```cpp
void func() {
    shared_ptr<Foo> fooPtr = make_shared<Foo>();
    ...
}
```

离开 func 作用域，fooPtr 就销毁了。

fooPtr 是唯一也是最后一个持有 foo 对象的智能指针。

所以离开 func 作用域时，其指向的 foo 对象就会销毁。

### 保存续命

```cpp
shared_ptr<Foo> globalPtr;

void func() {
    shared_ptr<Foo> fooPtr = make_shared<Foo>();
    ...
    globalPtr = fooPtr;
}
```

- 离开 func 作用域，fooPtr 就销毁了。
- 但是 globalPtr 是全局变量，直到程序退出才会销毁。
- 相当于帮原 fooPtr 指向的对象帮续命了！

### 提前释放

```cpp
void other() {
    globalPtr = nullptr;  // 相当于传统指针的 delete
}
```

但是如果现在又一个函数给 globalPtr 写入空指针。
这时之前对原对象的引用就没有了。

**对智能指针写入一个空指针可以使其指向的对象释放。**

对智能指针写入空指针的效果和 delete 很像，区别在于：

- 如果你忘了 delete 就完了！
- 你就算不写入空指针，智能指针也会自动释放，写入空指针只是把死期提前了一点而已……

```cpp
shared_ptr<Foo> p = make_shared<Foo>();

p = nullptr;  // 1
p.reset();    // 2
}             // 3
```

> P.S. 同理，vector 也可以通过 `v = {}` 或 `v.clear()` 来提前释放内存。

### 总结

- 当你需要分配一段内存空间：vector
- 当你需要创建单个对象：shared_ptr
- 当你想提前 delete：写入空指针

## 线程安全？

似乎很多三脚猫教材都在模棱两可地辩论一个问题：shared_ptr 到底是不是线程安全的？

不论什么类型，都要看你的用况，才能知道是不是线程安全，这里分为三种情况讨论：

1. 多个线程同时从同一个地方拷贝 shared_ptr 出来是安全的（多线程只读永远安全定律）：

```cpp
shared_ptr<T> a;
void t1() {
    shared_ptr<T> b1 = a;
}

void t1() {
    shared_ptr<T> b2 = a;
}
```

2. 多个线程同时从往同一个地方写入 shared_ptr 是不安全的（多线程 1 写 n 读定律）：

```cpp
shared_ptr<T> a;
void t1() {
    shared_ptr<T> b1;
    a = b1;
}

void t1() {
    shared_ptr<T> b2;
    a = b2;
}
```

> 这种情况下，你应该考虑的是 `atomic<shared_ptr<T>>`。

3. shared_ptr 并不保护其指向 T 类型的线程安全（你自己 T 实现的就不安全怪我指针???）：

```cpp
shared_ptr<T> a;
void t1() {
    a->b1 = 0;
}

void t1() {
    a->b1 = 1;
}
```

> 这种情况下，你应该考虑的是给你的 T 类型里面加个 mutex 保护好自己，而不是来怪我指针。

直接的答案：他们说的是，shared_ptr 的拷贝构造函数、析构函数是线程安全的，这不是废话吗？我只是拷贝另一个 shared_ptr，对那个 shared_ptr 又不进行更改，当然不会发生线程冲突咯。我自己析构关你其他 shared_ptr 什么事，当然就没有线程冲突咯。这是非常直观的，和普通指针的线程安全没有任何不同。

之所以这些狗币教材会辩论，是因为他们老爱多管闲事，他们了解到 shared_ptr 的底层细节中有个控制块的存在，而拷贝构造函数、析构函数需要修改控制块的计数值，所以实际标准库的实现中，会把这个计数器设为原子的，最终结果是使得 shared_ptr 在多线程中和普通指针一样安全。这是标准库底层实现细节，我们作为高层用户并不需要考虑他底层如何实现，我们只需要记住原始指针怎样用是线程安全的，shared_ptr 就怎样线程安全。

- 你会两个线程同时写入同一个原始指针吗？同样地，如果你原始指针不会犯错，shared_ptr 为什么会犯错？
- 你可以两个线程同时读取同一个全局的原始指针变量，同样地，shared_ptr 也可以，有任何区别吗？

反正，shared_ptr 内部专门为线程安全做过设计，你不用去操心。

## placement new

placement new 和 placement delete 也可以用 std::construct_at 和 std::destroy_at 代替：

```cpp
#include <new>

struct Foo {
    explicit Foo(int age) { ... }
    Foo(Foo &&) = delete;
    ~Foo() { ... }
};

void func() {
    alignas(Foo) unsigned char buffer[sizeof(Foo)];
    Foo *foo = std::construct_at(reinterpret_cast<Foo*>(buffer), 42, "hello"); // 等价于 new (buffer) Foo(42);
    ...
    std::destroy_at(foo); // 等价于 foo->~Foo();
}
```

> {{ icon.detail }} 在[内存模型专题](cpp_memory.md)中有进一步的详解。
