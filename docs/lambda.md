# 小彭老师带你学函数式编程

[TOC]

## 为什么需要函数？

```cpp
int main() {
    std::vector<int> a = {1, 2, 3, 4};
    int s = 0;
    for (int i = 0; i < a.size(); i++) {
        s += a[i];
    }
    fmt::println("sum = {}", s);
    return 0;
}
```

这是一个计算数组求和的简单程序。

但是，他只能计算数组 a 的求和，无法复用。

如果我们有另一个数组 b 也需要求和的话，就得把整个求和的 for 循环重新写一遍：

```cpp
int main() {
    std::vector<int> a = {1, 2, 3, 4};
    int s = 0;
    for (int i = 0; i < a.size(); i++) {
        s += a[i];
    }
    fmt::println("sum of a = {}", s);
    std::vector<int> b = {5, 6, 7, 8};
    s = 0;
    for (int i = 0; i < a.size(); i++) {
        s += b[i];
    }
    fmt::println("sum of b = {}", s);
    return 0;
}
```

这就出现了程序设计的大忌：代码重复。

> {{ icon.fun }} 例如，你有吹空调的需求，和充手机的需求。你为了满足这两个需求，购买了两台发电机，分别为空调和手机供电。第二天，你又产生了玩电脑需求，于是你又购买一台发电机，专为电脑供电……真是浪费！

重复的代码不仅影响代码的**可读性**，也增加了**维护**代码的成本。

+ 看起来乱糟糟的，信息密度低，让人一眼看不出代码在干什么的功能
+ 很容易写错，看走眼，难调试
+ 复制粘贴过程中，容易漏改，比如这里的 `s += b[i]` 可能写成 `s += a[i]` 而自己不发现
+ 改起来不方便，当我们的需求变更时，需要多处修改，比如当我需要改为计算乘积时，需要把两个地方都改成 `s *=`
+ 改了以后可能漏改一部分，留下 Bug 隐患
+ 敏捷开发需要反复修改代码，比如你正在调试 `+=` 和 `-=` 的区别，看结果变化，如果一次切换需要改多处，就影响了调试速度

### 狂想：没有函数的世界？

> {{ icon.story }} 如果你还是喜欢“一本道”写法的话，不妨想想看，完全不用任何标准库和第三方库的函数和类，把 `fmt::println` 和 `std::vector` 这些函数全部拆解成一个个系统调用。那这整个程序会有多难写？

```cpp
int main() {
#ifdef _WIN32
    int *a = (int *)VirtualAlloc(NULL, 4096, MEM_COMMIT, PAGE_EXECUTE_READWRITE);
#else
    int *a = (int *)mmap(NULL, 4 * sizeof(int), PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
#endif
    a[0] = 1;
    a[1] = 2;
    a[2] = 3;
    a[3] = 4;
    int s = 0;
    for (int i = 0; i < 4; i++) {
        s += a[i];
    }
    char buffer[64];
    buffer[0] = 's';
    buffer[1] = 'u';
    buffer[2] = 'm';
    buffer[3] = ' ';
    buffer[4] = '=';
    buffer[5] = ' '; // 例如，如果要修改此处的提示文本，甚至需要修改后面的 len 变量...
    int len = 6;
    int x = s;
    do {
        buffer[len++] = '0' + x % 10;
        x /= 10;
    } while (x);
    buffer[len++] = '\n';
#ifdef _WIN32
    WriteFile(GetStdHandle(STD_OUTPUT_HANDLE), buffer, len, NULL, NULL);
#else
    write(1, buffer, len);
#endif
    int *b = (int *)a;
    b[0] = 4;
    b[1] = 5;
    b[2] = 6;
    b[3] = 7;
    int s = 0;
    for (int i = 0; i < 4; i++) {
        s += b[i];
    }
    len = 6;
    x = s;
    do {
        buffer[len++] = '0' + x % 10;
        x /= 10;
    } while (x);
    buffer[len++] = '\n';
#ifdef _WIN32
    WriteFile(GetStdHandle(STD_OUTPUT_HANDLE), buffer, len, NULL, NULL);
#else
    write(1, buffer, len);
#endif
#ifdef _WIN32
    VirtualFree(a, 0, MEM_RELEASE);
#else
    munmap(a);
#endif
    return 0;
}
```

不仅完全没有可读性、可维护性，甚至都没有可移植性。

除非你只写应付导师的“一次性”程序，一旦要实现复杂的业务需求，不可避免的要自己封装函数或类。网上所有鼓吹“不封装”“设计模式是面子工程”的反智言论，都是没有做过大型项目的。


### 设计模式追求的是“可改”而不是“可读”！

很多设计模式教材片面强调**可读性**，仿佛设计模式就是为了“优雅”“高大上”“美学”？使得很多人认为，“我这个是自己的项目，不用美化给领导看”而拒绝设计模式。实际上设计模式的主要价值在于**方便后续修改**！

> {{ icon.fun }} 例如 B 站以前只支持上传普通视频，现在叔叔突然提出：要支持互动视频，充电视频，视频合集，还废除了视频分 p，还要支持上传短视频，竖屏开关等……每一个叔叔的要求，都需要大量程序员修改代码，无论涉及前端还是后端。

与建筑、绘画等领域不同，一次交付完毕就可以几乎永久使用。而软件开发是一个持续的过程，每次需求变更，都导致代码需要修改。开发人员几乎需要一直围绕着软件代码，不断的修改。调查表明，程序员 90% 的时间花在**改代码**上，**写代码**只占 10%。

> {{ icon.fun }} 软件就像生物，要不断进化，软件不更新不维护了等于死。如果一个软件逐渐变得臃肿难以修改，无法适应新需求，那他就像已经失去进化能力的生物种群，如《三体》世界观中“安顿”到澳大利亚保留区里“绝育”的人类，被淘汰只是时间问题。

如果我们能在**写代码**阶段，就把程序准备得**易于后续修改**，那就可以在后续 90% 的**改代码**阶段省下无数时间。

如何让代码易于修改？前人总结出一系列常用的写法，这类写法有助于让后续修改更容易，各自适用于不同的场合，这就是设计模式。

提升可维护性最基础的一点，就是避免重复！

当你有很多地方出现重复的代码时，一旦需要涉及修改这部分逻辑时，就需要到每一个出现了这个逻辑的代码中，去逐一修改。

> {{ icon.fun }} 例如你的名字，在出生证，身份证，学生证，毕业证，房产证，驾驶证，各种地方都出现了。那么你要改名的话，所有这些证件都需要重新印刷！如果能把他们合并成一个“统一证”，那么只需要修改“统一证”上的名字就行了。

不过，现实中并没有频繁改名字的需求，这说明：

- 对于不常修改的东西，可以容忍一定的重复。
- 越是未来有可能修改的，就越需要设计模式降重！

例如数学常数 PI = 3.1415926535897，这辈子都不可能出现修改的需求，那写死也没关系。如果要把 PI 定义成宏，只是出于“记不住”“写起来太长了”“复制粘贴麻烦”。所以对于 PI 这种不会修改的东西，降重只是增加**可读性**，而不是**可修改性**。

> {{ icon.tip }} 但是，不要想当然！需求的千变万化总是超出你的想象。

例如你做了一个“愤怒的小鸟”游戏，需要用到重力加速度 g = 9.8，你想当然认为 g 以后不可能修改。老板也信誓旦旦向你保证：“没事，重力加速度不会改变。”你就写死在代码里了。

没想到，“愤怒的小鸟”老板突然要求你加入“月球章”关卡，在这些关卡中，重力加速度是 g = 1.6。

如果你一开始就已经把 g 提取出来，定义为常量：

```cpp
struct Level {
    const double g = 9.8;

    void physics_sim() {
        bird.v = g * t; // 假装这里是物理仿真程序
        pig.v = g * t;  // 假装这里是物理仿真程序
    }
};
```

那么要支持月球关卡，只需修改一处就可以了。

```cpp
struct Level {
    double g;

    Level(Chapter chapter) {
        if (chapter == ChapterMoon) {
            g = 1.6;
        } else {
            g = 9.8;
        }
    }

    void physics_sim() {
        bird.v = g * t; // 无需任何修改，自动适应了新的非常数 g
        pig.v = g * t;  // 无需任何修改，自动适应了新的非常数 g
    }
};
```

> {{ icon.fun }} 小彭老师之前做 zeno 时，询问要不要把渲染管线节点化，方便用户动态编程？张猩猩就是信誓旦旦道：“渲染是一个高度成熟领域，不会有多少修改需求的。”小彭老师遂写死了渲染管线，专为性能极度优化，几个月后，张猩猩羞答答找到小彭老师：“小彭老师，那个，渲染，能不能改成节点啊……”。这个故事告诉我们，甲方的信誓旦旦放的一个屁都不能信。

### 用函数封装

函数就是来帮你解决代码重复问题的！要领：

**把共同的部分提取出来，把不同的部分作为参数传入。**

```cpp
void sum(std::vector<int> const &v) {
    int s = 0;
    for (int i = 0; i < v.size(); i++) {
        s += v[i];
    }
    fmt::println("sum of v = {}", s);
}

int main() {
    std::vector<int> a = {1, 2, 3, 4};
    sum(a);
    std::vector<int> b = {5, 6, 7, 8};
    sum(b);
    return 0;
}
```

这样 main 函数里就可以只关心要求和的数组，而不用关心求和具体是如何实现的了。事后我们可以随时把 sum 的内容偷偷换掉，换成并行的算法，main 也不用知道。这就是**封装**，可以把重复的公共部分抽取出来，方便以后修改代码。

> {{ icon.fun }} sum 函数相当于，当需要吹空调时，插上空调插座。当需要给手机充电时，插上手机充电器。你不需要关心插座里的电哪里来，“国家电网”会替你想办法解决，想办法优化，想办法升级到绿色能源。你只需要吹着空调给你正在开发的手机 App 优化就行了，大大减轻程序员心智负担。

### 要封装，但不要耦合

但是！这段代码仍然有个问题，我们把 sum 求和的结果，直接在 sum 里打印了出来。sum 里写死了，求完和之后只能直接打印，调用者 main 根本无法控制。

这是一种错误的封装，或者说，封装过头了。

> {{ icon.fun }} 你把手机充电器 (fmt::println) 焊死在了插座 (sum) 上，现在这个插座只能给手机充电 (用于直接打印) 了，不能给笔记本电脑充电 (求和结果不直接用于打印) 了！尽管通过更换充电线 (参数 v)，还可以支持支持安卓 (a) 和苹果 (b) 两种手机的充电，但这样焊死的插座已经和笔记本电脑无缘了。

### 每个函数应该职责单一，别一心多用

很明显，“打印”和“求和”是两个独立的操作，不应该焊死在一块。

sum 函数的本职工作是“数组求和”，不应该附赠打印功能。

sum 计算出求和结果后，直接 return 即可。

> {{ icon.fun }} 如何处理这个结果，是调用者 main 的事，正如“国家电网”不会管你用他提供的电来吹空调还是玩游戏一样，只要不妨碍到其他居民的正常用电。

```cpp
int sum(std::vector<int> const &v) {
    int s = 0;
    for (int i = 0; i < v.size(); i++) {
        s += v[i];
    }
    return s;
}

int main() {
    std::vector<int> a = {1, 2, 3, 4};
    fmt::println("sum of a = {}", sum(a));
    std::vector<int> b = {5, 6, 7, 8};
    fmt::println("sum of b = {}", sum(b));
    return 0;
}
```

这就是设计模式所说的**职责单一原则**。

### 二次封装

假设我们要计算一个数组的平均值，可以再定义个函数 average，他可以基于 sum 实现：

```cpp
int sum(std::vector<int> const &v) {
    int s = 0;
    for (int i = 0; i < v.size(); i++) {
        s += v[i];
    }
    return s;
}

double average(std::vector<int> const &v) {
    return (double)sum(v) / v.size();
}

int main() {
    std::vector<int> a = {1, 2, 3, 4};
    fmt::println("average of a = {}", average(a));
    std::vector<int> b = {5, 6, 7, 8};
    fmt::println("average of b = {}", average(b));
    return 0;
}
```

进一步封装一个打印数组所有统计学信息的函数：

```cpp
void print_statistics(std::vector<int> const &v) {
    if (v.empty()) {
        fmt::println("this is empty...");
    } else {
        fmt::println("sum: {}", sum(v));
        fmt::println("average: {}", average(v));
        fmt::println("min: {}", min(v));
        fmt::println("max: {}", max(v));
    }
}

int main() {
    std::vector<int> a = {1, 2, 3, 4};
    print_statistics(a);
    std::vector<int> b = {5, 6, 7, 8};
    print_statistics(b);
    return 0;
}
```

暴露 API 时，要同时提供底层的 API 和高层封装的 API。用户如果想要控制更多细节可以调用底层 API，想要省事的用户可以调用高层封装好的 API。

> {{ icon.tip }} 高层封装 API 应当可以完全通过调用底层 API 实现，提供高层 API 只是方便初级用户使用和理解。

> {{ icon.story }}
    例如 `libcurl` 就提供了 `curl_easy` 和 `curl_multi` 两套 API。

    - `curl_multi` 提供了超详细的参数，把每个操作分拆成多步，方便用户插手细节，满足高级用户的定制化需求，但太过复杂，难以学习。
    - `curl_easy` 是对 `curl_multi` 的再封装，提供了更简单的 API，但是对具体细节就难以操控了，适合初学者上手。

### Linus 的最佳实践：每个函数不要超过 3 层嵌套，一行不要超过 80 字符，每个函数体不要超过 24 行

Linux 内核为什么坚持使用 8 缩进为代码风格？

因为高缩进可以避免程序员写出嵌套层数太深的代码，当他写出太深嵌套时，巨大的 8 缩进会让代码变得非常偏右，写不下多少空间。从而让程序员自己红着脸“对不起，我把单个函数写太深了”然后赶紧拆分出多个函数来。

此外，他还规定了单一一个函数必须在终端宽度 80 x 24 中显示得下，否则就需要拆分成多个函数重写，这配合 8 缩进，有效的限制了嵌套的层数，迫使程序员不得不重新思考，更解耦的写法出来。

## 为什么需要函数式？

你产生了两个需求，分别封装了两个函数：

- `sum` 求所有元素的和
- `product` 求所有元素的积

```cpp
int sum(std::vector<int> const &v) {
    int ret = v[0];
    for (int i = 1; i < v.size(); i++) {
        ret += v[i];
    }
    return ret;
}

int product(std::vector<int> const &v) {
    int ret = v[0];
for (int i = 1; i < v.size(); i++) {
        ret *= v[i];
    }
    return ret;
}

int main() {
    std::vector<int> a = {1, 2, 3, 4};
    fmt::println("sum: {}", sum(a));
    fmt::println("product: {}", product(a));
    return 0;
}
```

注意到 `sum` 和 `product` 的内容几乎如出一辙，唯一的区别在于：

- `sum` 的循环体为 `+=`；
- `product` 的循环体为 `*=`。

这种函数体内有部分代码重复，但又有特定部分不同，难以抽离。

该怎么复用这重复的部分代码呢？

我们要把 `sum` 和 `product` 合并成一个函数 `generic_sum`。然后通过函数参数，把差异部分（0、`+=`）“注入”到两个函数原本不同地方。

### 枚举的糟糕用法

如何表示我这个函数是要做求和 `+=` 还是求积 `*=`？

让我们定义枚举：

```cpp
enum Mode {
    ADD, // 求和操作
    MUL, // 求积操作
};

int generic_sum(std::vector<int> const &v, Mode mode) {
    int ret = v[0];
    for (int i = 1; i < v.size(); i++) {
        if (mode == ADD) { // 函数内判断枚举，决定要做什么操作
            ret += v[i];
        } else if (mode == MUL) {
            ret *= v[i];
        }
    }
    return ret;
}

int main() {
    std::vector<int> a = {1, 2, 3, 4};
    fmt::println("sum: {}", generic_sum(a, ADD)); // 用户指定他想要的操作
    fmt::println("product: {}", generic_sum(a, MUL));
    return 0;
}
```

然而，如果用户现在想要求数组的**最大值**呢？

枚举中还没有实现最大值的操作……要支持，就得手忙脚乱地去修改 `generic_sum` 函数和 `Mode` 枚举原本的定义，真麻烦！

```cpp
enum Mode {
    ADD,
    MUL,
    MAX, // ***改***
};

int generic_sum(std::vector<int> const &v, Mode mode) {
    int ret = v[0];
    for (int i = 1; i < v.size(); i++) {
        if (mode == ADD) {
            ret += v[i];
        } else if (mode == MUL) {
            ret *= v[i];
        } else if (mode == MAX) { // ***改***
            ret = std::max(ret, v[i]); // ***改***
        }
    }
    return ret;
}

int main() {
    std::vector<int> a = {1, 2, 3, 4};
    generic_sum(a, MAX); // ***改***
    return 0;
}
```

> {{ icon.tip }} 我用 `// ***改***` 指示了所有需要改动的地方。

为了增加一个求最大值的操作，就需要三处分散在各地的改动！

不仅如此，还容易抄漏，抄错，比如 `MAX` 不小心打错成 `MUL` 了，自己却没发现，留下 BUG 隐患。

这样写代码的方式，心智负担极大，整天就提心吊胆着东一块，西一块的散装代码，担心着有没有哪个地方写错写漏，严重妨碍了开发效率。

并且写出来的代码也不能适应需求的变化：假如我需要支持 `MIN` 呢？又得改三个地方！这违背了设计模式的**开闭原则**。

* 开闭原则: 对扩展开放，对修改封闭。指的是软件在适应需求变化时，应尽量通过**扩展代码*来实现变化，而不是通过*修改已有代码**来实现变化。

使用枚举和 if-else 实现多态，难以扩展，还要一直去修改原函数的底层实现，就违背了**开闭原则**。

### 函数式编程光荣救场

如果我们可以“注入”代码就好了！能否把一段“代码”作为 `generic_sum` 函数的参数呢？

代码，实际上就是函数，注入代码就是注入函数。我们先定义出三个不同操作对应的函数：

```cpp
int add(int a, int b) {
    return a + b;
}

int mul(int a, int b) {
    return a * b;
}

int max(int a, int b) {
    return std::max(a, b);
}
```

然后，把这三个小函数，作为另一个大函数 `generic_sum` 的参数就行！

```cpp
int generic_sum(std::vector<int> const &v, auto op) {
    int ret = v[0];
    for (int i = 1; i < v.size(); i++) {
        // 函数作者无需了解用户指定的“操作”具体是什么
        // 只需要调用这一“操作”，得到结果就行
        ret = op(ret, v[i]);
    }
    return ret;
}

int main() {
    std::vector<int> a = {1, 2, 3, 4};
    // 用户无需关心函数的具体实现是什么
    // 只需随心所欲指定他的“操作”作为参数
    generic_sum(a, add);
    generic_sum(a, product);
    generic_sum(a, max);
    return 0;
}
```

责任明确了，我们成功把一部分细节从 `generic_sum` 中进一步抽离。

- 库作者 `generic_sum` 不必了解 `main` 的操作具体是什么，他只负责利用这个操作求“和”。
- 库用户 `main` 不必了解 `generic_sum` 如何实现操作累加，他只管注入“如何操作”的代码，以函数的形式。

### 我用了 C++20 的函数参数 auto 语法糖

```cpp
int generic_sum(std::vector<int> const &v, auto op) {
}
```

这里的参数 op 类型声明为 auto，效果就是，op 这个参数现在能接受任意类型的对象了（包括函数！）

```cpp
int generic_sum(std::vector<int> const &v, auto op) {
    ...
}
```

> {{ icon.detail }} 准确的说，`auto op` 参数的效果是使 `generic_sum` 变为一个**模板函数**，其中 op 参数变成了模板参数，能够接受任意类型了。而写明类型的参数 `std::vector<int> const &v` 就没有任何额外效果，就只能接受 `vector<int>` 而已。

如果你不支持 C++20 的话，需要显式写出 `template`，才能实现同样的效果：

```cpp
template <typename Op>
int generic_sum(std::vector<int> const &v, Op op) {
    ...
}
```

> {{ icon.fun }} C++11：auto 只能用于定义变量以及作为函数返回类型的占位符（无法自行推导）；C++14：函数返回类型可以是 auto 并自动推导；C++17：模板非类型参数也可以 auto；C++20：函数参数也可以是 auto 了；（狂想）C++47：auto 现在是 C++47 的唯一关键字，用户只需不断输入 auto-auto-auto，编译器内建人工智能自动识别你的意图生成机器码。

### 函数也是对象！

在过去的**面向对象编程范式*中，函数（代码）和对象（数据）被*割裂*开来，他们愚昧地认为*函数不是对象**。

**函数式编程范式*则认为：*函数也是一种变量，函数可以作为另一个函数的参数！**

> {{ icon.fun }} Function lives matter!

> {{ icon.detail }} 面向对象就好比计算机的“哈佛架构”，代码和数据割裂，代码只能单方面操作数据。函数式就好比“冯诺依曼架构”，代码也是数据。看似会导致低效，实则大大方便了动态加载新程序，因而现在的计算机基本都采用了“冯诺依曼架构”。

总之，函数也是对象，被亲切地尊称为**函数对象**。

### C++11 引入 Lambda 语法糖

C++98 时代，人们还需要单独跑到 `main` 外面，专门定义 `add`、`mul`、`max` 函数。弄得整个代码乱哄哄的，非常麻烦。

```cpp
int add(int a, int b) {
    return a + b;
}

int mul(int a, int b) {
    return a * b;
}

int max(int a, int b) {
    return std::max(a, b);
}

int main() {
    std::vector<int> a = {1, 2, 3, 4};
    generic_sum(a, add);
    generic_sum(a, product);
    generic_sum(a, max);
    return 0;
}
```

C++11 引入了 *Lambda 表达式*语法，允许你就地创建一个函数。

```cpp
int main() {
    std::vector<int> a = {1, 2, 3, 4};

    auto add = [](int a, int b) {
        return a + b;
    };
    auto mul = [](int a, int b) {
        return a * b;
    };
    auto max = [](int a, int b) {
        return std::max(a, b);
    };

    generic_sum(a, add);
    generic_sum(a, product);
    generic_sum(a, max);
    return 0;
}
```

不用往 `main` 外面塞垃圾了，一清爽。

更进一步，我们甚至不用定义变量，直接把 Lambda 表达式写在 `generic_sum` 的参数里就行了！

```cpp
int main() {
    std::vector<int> a = {1, 2, 3, 4};

    generic_sum(a, [](int a, int b) {
        return a + b;
    });
    generic_sum(a, [](int a, int b) {
        return a * b;
    });
    generic_sum(a, [](int a, int b) {
        return std::max(a, b);
    }); // ***改***
    return 0;
}
```

> {{ icon.tip }} 以上写法都是等价的。

要支持一个新操作，只需修改一处地方：在调用 `generic_sum` 时就地创建一个函数。随叫随到，不用纠结于“起名强迫症”，是不是很方便呢？

> {{ icon.detail }} 准确的说，Lambda 创建的是函数对象 (function object) 或称仿函数 (functor) 而不是传统意义上的函数。

> {{ icon.story }} 其实 C++98 时代人们就已经大量在用 `operator()()` 模拟函数对象了，著名的第三方库 Boost 也封装了各种函数式常用的容器和工具。C++11 才终于把**函数对象**这个概念转正，并引入了更方便的 Lambda 语法糖。

> {{ icon.fun }} 即使是面向对象的头号孝子 Java，也已经开始引入函数式的 Lambda 语法糖，C\# 的 LINQ 更是明目张胆的致敬 map-reduce 全家桶，甚至 C 语言用户也开始玩各种函数指针回调……没办法，函数式确实方便呀！

### 依赖注入原则

函数对象 `op` 作为参数传入，让 `generic_sum` 内部去调用，就像往 `generic_sum` 体内“注入”了一段自定义代码一样。

这可以让 `generic_sum` 在不修改本体的情况下，通过修改“注入”部分，轻松扩展，满足**开闭原则**。

更准确的说，这体现的是设计模式所要求的**依赖注入原则**。

* 依赖注入原则: 一个封装好的函数或类，应该尽量依赖于抽象接口，而不是依赖于具体实现。这可以提高程序的灵活性和可扩展性。

四大编程范式都各自发展出了**依赖注入原则**的解决方案：

- 面向过程编程范式中，**函数指针**就是那个抽象接口。
- 面向对象编程范式中，**虚函数**就是那个抽象接口。
- 函数式编程范式中，**函数对象**就是那个抽象接口。
- 模板元编程范式中，**模板参数**就是那个抽象接口。

同样是把抽象接口作为参数，同样解决可扩展问题。

函数指针贴近底层硬件，虚函数方便整合多个接口，函数对象轻量级、随地取用，模板元有助高性能优化，不同的编程范式殊途同归。

### 低耦合，高内聚

依赖注入原则可以减少代码之间的耦合度，大大提高代码的灵活性和可扩展性。

* 耦合度: 指的是一个模块、类、函数和其他模块、类、函数之间的关联程度。耦合度越低，越容易进行单元测试、重构、复用和扩展。

> {{ icon.fun }} 高耦合度的典型是“牵一发而动全身”。低耦合的典范是蚯蚓，因为蚯蚓可以在任意断面切开，还能活下来，看来蚯蚓的身体设计非常“模块化”呢。

通常来说，软件应当追求低耦合度，适度解耦的软件能更快适应需求变化。但过度的低耦合也会导致代码过于分散，不易阅读和修改，甚至可能起到反效果。

> {{ icon.tip }} 若你解耦后，每次需求变化要改动的地方变少了，那就是合理的解耦。若你过分解耦，代码东一块西一块，以至于需求变化时需要到处改，比不解耦时浪费的时间还要多，那就是解耦过度。

> {{ icon.fun }} 完全零耦合的程序每个函数互不联系，就像把蚯蚓拆散成一个个独立的细胞一样。连初始需求“活着”都实现不了，谈何适应需求变化？所以解耦也切勿矫枉过正。

为了避免解耦矫枉过正，人们又提出了内聚的概念，并规定解耦的前提是：不耽误内聚。耽误到内聚的解耦，就只会起到降低可维护性的反效果了。

* 内聚: 指的是同一个模块、类、函数内部各个元素之间的关联程度。内聚度越高，功能越独立，越方便集中维护。

> {{ icon.fun }} 例如，人的心脏专门负责泵血，肝脏只负责解毒，这就是高内聚的人体器官。若人的心脏还要兼职解毒，肝脏还兼职泵血，看似好像是增加了“万一心脏坏掉”的冗余性，实际上把“泵血”这一功能拆散到各地，无法“集中力量泵大血”了。

> {{ icon.detail }} 人类的大脑和 CPU 一样，也有“缓存局域性 (cache-locality)”的限制：不能同时在很多个主题之间快速切换，无论是时间上的还是空间上的割裂 (cache-miss)，都会干扰程序员思维的连贯性，从而增大心智负担。

好的软件要保持低耦合，同时高内聚。

> {{ icon.fun }} 就像“民主集中制”一样，既要监督防止大权独揽，又要集中力量办一个人办不成的大事。

### 与传统面向对象的对比

传统的面向对象同样可以用**虚函数接口类*模拟*函数对象**一样的功能，只不过没有 lambda 和闭包的语法加持，写起来非常繁琐，就和在 C 语言里“模拟”面向对象一样。

> {{ icon.fun }} 为了这么小的一个代码块，单独定义一个类，就像妈妈开一架“空中战车” A380 只是为了接你放学一样，等你值好机的时间我自己走都走到了。而函数式中，用 lambda 就地定义函数对象，相当于随地抓来一台共享单车开走。

```cpp
struct OpBase { // 面向对象：遇事不决先定义接口……
    virtual int compute(int a, int b) = 0;
    virtual ~OpBase() = default;
};

struct OpAdd : OpBase {
    int compute(int a, int b) override {
        return a + b;
    }
};

struct OpMul : OpBase {
    int compute(int a, int b) override {
        return a * b;
    }
};

struct OpMax : OpBase {
    int compute(int a, int b) override {
        return std::max(a, b);
    }
};

int generic_sum(std::vector<int> const &v, OpBase *op) {
    int ret = v[0];
    for (int i = 1; i < v.size(); ++i) {
        ret = op->compute(ret, v[i]); // 写起来也麻烦，需要调用他的成员函数，成员函数又要起名……
    }
    delete op;
    return ret;
}

int main() {
    std::vector<int> a = {1, 2, 3, 4};

    generic_sum(a, new OpAdd());
    generic_sum(a, new OpMul());
    generic_sum(a, new OpMax());
    return 0;
}
```

不仅需要定义一堆类，接口类，实现类，继承来继承去，还需要管理讨厌的指针，代码量翻倍，没什么可读性，又影响运行效率。

> {{ icon.fun }} 3 年 2 班小彭同学，你的妈妈开着 A380 来接你了。

而现代 C++ 只需 Lambda 语法就地定义函数对象，爽。

```cpp
    generic_sum(a, [](int a, int b) {
        return a + b;
    });
    generic_sum(a, [](int a, int b) {
        return a * b;
    });
    generic_sum(a, [](int a, int b) {
        return std::max(a, b);
    });
```

### 函数对象在模板加持下静态分发

刚刚，我们的实现用了 `auto op` 做参数，这等价于让 `generic_sum` 变成一个模板函数。

```cpp
int generic_sum(std::vector<int> const &v, auto op);

// 不支持 C++20 时的替代写法：
template <typename Op>
int generic_sum(std::vector<int> const &v, Op op);
```

这意味着每当用户指定一个新的函数对象（lambda）时，`generic_sum` 都会重新实例化一遍。

```cpp
    generic_sum(a, [](int a, int b) {
        return a + b;
    });
    generic_sum(a, [](int a, int b) {
        return a * b;
    });
    generic_sum(a, [](int a, int b) {
        return std::max(a, b);
    });
```

编译后，会变成类似于这样：

```cpp
    generic_sum<add>(a);
    generic_sum<mul>(a);
    generic_sum<max>(a);
```

会生成三份函数，每个都是独立编译的：

```cpp
int generic_sum<add>(std::vector<int> const &v) {
    int ret = v[0];
    for (int i = 1; i < v.size(); ++i) {
        ret = add(ret, v[i]);
    }
    return ret;
}
int generic_sum<mul>(std::vector<int> const &v) {
    int ret = v[0];
    for (int i = 1; i < v.size(); ++i) {
        ret = mul(ret, v[i]);
    }
    return ret;
}
int generic_sum<max>(std::vector<int> const &v) {
    int ret = v[0];
    for (int i = 1; i < v.size(); ++i) {
        ret = max(ret, v[i]);
    }
    return ret;
}
```

这允许编译器为每个版本的 `generic_sum` 单独做优化，量身定制最优的代码。

例如 `add` 这个函数对象，因为只在 `generic_sum<add>` 中使用了，会被被编译器自动内联，不会产生函数调用和跳转的指令，各自优化成单独一条加法 / 乘法 / 最大值指令等。

> {{ icon.detail }} 比如，编译器会检测到 `+=` 可以矢量化，于是用 `_mm_add_epi32` 替代了。同理，mul 则用 `_mm_mullo_epi32` 替代，max 则用 `_mm_max_epi32` 替代等，各自分别生成了各自版本最优的代码。而如果是普通的函数指针，不会生成三份量身定做的实例，无法矢量化（有一种例外，就是编译器检测到了 `generic_sum` 似乎只有这三种可能参数，然后做了 IPO 优化，但并不如模板实例化一样稳定强制）。

为三种不同的 op 参数分别定做三份。虽然增加了编译时间，膨胀了生成的二进制体积；但生成的机器码是分别针对每种特例一对一深度优化的，更高效。

> {{ icon.story }} 例如矩阵乘法（gemm）的最优算法，对于不同的矩阵大小和形状是不同的。著名的线性代数库 CUBLAS 和 MKL 中，会自动根据用户输入的矩阵形状，选取最优的算法。也就是说，CUBLAS 库里其实存着适合各种矩阵大小排列组合的算法代码（以 fatbin 格式存储在二进制中）。当调用矩阵乘法时，自动查到最适合的一版来调用给你。类似 gemm，还有 gemv、spmv……所有的矩阵运算 API 都经历了这样的“编译期”暴力排列组合，只为“运行时”释放最大性能！这也导致编译好的 cublas.dll 文件来到了恐怖的 20 MB 左右，而我们称之为高效。

### 函数对象也可在 function 容器中动态分发

Lambda 函数对象的类型是匿名的，每个 Lambda 表达式都会创建一个全新的函数对象类型，这使得 `generic_sum` 对于每个不同的 Lambda 都会实例化一遍。虽然有利于性能优化，但也影响了编译速度和灵活性。

> {{ icon.detail }} 通常，我们只能通过 `decltype(add)` 获取 `add` 这个 Lambda 对象的类型。也只能通过 `auto` 来捕获 Lambda 对象为变量。

为此，标准库提供了 `std::function` 容器，他能容纳任何函数对象！无论是匿名的 Lambda 函数对象，还是普普通通的函数指针，都能纳入 `std::function` 的体内。

唯一的代价是，你需要指定出所有参数的类型，和返回值的类型。

例如参数为两个 `int`，返回 `int` 的函数，可以用 `std::function<int(int, int)>` 容器存储。

```cpp
auto add_lambda = [](int a, int b) { // Lambda 函数对象
    return a + b;
};

struct AddClass {
    int operator()(int a, int b) {   // 自定义类模拟函数对象
        return a + b;
    }
};
AddClass add_object;

int add_regular_func(int a, int b) { // 普通函数
    return a + b;
}

std::function<int(int, int)> add; // 所有广义函数对象，统统接纳
add = add_lambda;           // OK
add = add_object;           // OK
add = add_regular_func;     // OK
```

```cpp
int generic_sum(std::vector<int> const &v,
                std::function<int(int, int)> op) {
    int ret = v[0];
    for (int i = 1; i < v.size(); ++i) {
        ret = op(ret, v[i]); // 写起来和模板传参时一样无感
    }
    // 无需指针，无需 delete，function 能自动管理函数对象生命周期
    return ret;
}
```

> {{ icon.detail }} 如果还想支持任意类型的参数和返回值，那么你可以试试看 `std::function<std::any(std::any)>`。这里 `std::any` 是个超级万能容器，可以容纳任何对象，他和 `std::function` 一样都采用了“类型擦除 (type-erasure)”技术，缺点是必须配合 `std::any_cast` 才能取出使用，之后的模板元进阶专题中会详细介绍他们的原理，并带你自己做一个擦加法的类型擦除容器。

函数式编程，能在静态与动态之间轻松切换，**高性能*与*灵活性**任君选择。

- 在需要性能的**瓶颈代码**中用模板传参，编译期静态分发，多次量身定做，提高运行时性能。

* 瓶颈代码: 往往一个程序 80% 的时间花在 20% 的代码上。这 20% 是在程序中频繁执行的、计算量大的、或者调用特别耗时的函数。针对这部分瓶颈代码优化即可，而剩余的 80% 打酱油代码，大可以怎么方便怎么写。

- 在性能无关紧要的顶层业务逻辑中用 function 容器传参，运行时动态分发，节省编译体积，方便持久存储，灵活易用。

> {{ icon.tip }} 例如上面的 `generic_sum` 函数，如果我们突然想要高性能了，只需把 `std::function<int(int, int)> op` 轻轻改为 `auto op` 就轻松切换到静态分发模式了。

而虚函数一旦用了，基本就只能动态分发了，即使能被 IPO 优化掉，虚表指针也永远占据着一个 8 字节的空间，且永远只能以指针形式传来传去。

> {{ icon.detail }} 一种静态分发版的虚函数替代品是 CRTP，他基于模板元编程，但与虚函数之间切换困难，不像函数对象那么无感，之后的模板元专题课中会专门介绍。

### 案例：函数对象的动态分发用于多线程任务队列

主线程不断地向工作者线程发送函数对象，令其代为执行：

```cpp
mt_queue<std::function<void()>> task_queue;

void main_thread() {
    task_queue.push([] {
        fmt::println("正在执行任务1");
    });
    task_queue.push([] {
        fmt::println("正在执行任务2");
    });
}

void worker_thread() {
    while (true) {
        auto task = task_queue.pop();
        task();
    }
}
```

> {{ icon.detail }} `mt_queue` 是小彭老师封装的多线程安全的消息队列，实现原理会在稍后的多线程专题课中详细讲解。

### 函数对象的重要机制：闭包

闭包是函数对象的重要机制，他允许函数对象捕获外部变量，并在函数对象内部使用这些变量。

```cpp
int x = 10;
auto add_x = [x](int a) {
    return a + x;
};
fmt::println("{}", add_x(5)); // 输出 15
```

> {{ icon.tip }} 闭包捕获的变量默认是只读的，如果需要修改捕获的变量，可以加上 `mutable` 修饰，见后文。

#### 闭包的本质是语法糖

Lambda 函数对象的闭包语法：

```cpp
int x = 10;
auto add_x = [x](int a) {
    return a + x;
};
```

实际上等价于一个带有 `operator()` 成员函数的结构体：

```cpp
struct Lambda {
    int x;
    Lambda(int val) : x(val) {}

    int operator() (int a) const {
        return a + x;
    }
};

int main() {
    int x = 10;
    Lambda add_x(x);
    fmt::println("{}", add_x(5)); // 输出 15
    return 0;
}
```

> {{ icon.tip }} 相当于我们写的 lambda 函数体，实际上被编译器移到了 `Lambda` 类的 `operator()` 成员函数体内。

而且这结构体是匿名的，没有确定的名字，此处类名 `Lambda` 只是示意，因而平时只能通过 `auto` 保存即时创建的 lambda 对象。

**而所谓的闭包捕获变量，实际上就是这个结构体的成员！**

按值捕获，就相当于结构体成员里拷贝了一份同名的成员；如果是引用捕获，就相当于结构体里的成员是个引用。

> {{ icon.tip }} 可以在 https://cppinsights.io 这个网站，自动拆解包括 Lambda 在内的所有现代 C++ 语法糖为原始的结构体和函数。更多好用的工具网站可以看我们 [工具和项目推荐](recommend.md) 专题章节。

对于引用，则是等价于结构体成员中含有一份引用作为成员：

```cpp
int x = 10;
auto inc_x = [&x](int a) {
    return x++;
};
```

```cpp
struct Lambda {
    int &x;
    Lambda(int &val) : x(val) {}

    int operator() () const {
        return x++;
    }
};

int main() {
    int x = 10;
    Lambda inc_x(x);
    fmt::println("{}", inc_x()); // 输出 10
    fmt::println("{}", inc_x()); // 输出 11
    fmt::println("{}", inc_x()); // 输出 12
    fmt::println("{}", x);       // 输出 13
    return 0;
}
```

#### `operator()` 很有迷惑性

匿名 lambda 对象：

```cpp
auto lambda = [] (int a) {
    return a + 1;
};
int ret = lambda(2);
```

等价于以下的类：

```cpp
struct Lambda {
    int operator() (int a) const {
        return a + 1;
    }
};
Lambda lambda;
int ret = lambda(2);
```

很多同学都分不清 `operator` `operator()` `opeartor()()`，这个括号确实很有迷惑性，今天我来解释一下。

你现在，把上面这段代码，改成这样：

```cpp
struct Lambda {
    int call (int a) const {
        return a + 1;
    }
};
Lambda lambda;
int ret = lambda.call(2);
```

是不是很容易看懂？这就是定义了一个成员函数 `call`，然后调用这个成员函数。

现在，进一步改成：

```cpp
struct Lambda {
    int operator_call (int a) const {
        return a + 1;
    }
};
Lambda lambda;
int ret = lambda.operator_call(2);
```

能不能理解？这就是把函数名改成了 `operator_call`，依然是一个成员函数。

重点来了，我们把函数名，注意是函数名叫 `operator()`，这个空的圆括号是函数名的一部分！

```cpp
struct Lambda {
    int operator() (int a) const {
        return a + 1;
    }
};
Lambda lambda;
int ret = lambda.operator() (2);
```

能不能理解？`operator` 是一个特殊的关键字，效果是和后面的一个运算符结合，形成一个特殊的“标识符”，这个“标识符”和普通函数名一样，都是“单个单词”，不可分割。

例如 `operator+` 就是一个标识符，`operator[]` 也是一个标识符，我们这里的 `operator()` 也是一个标识符，没有什么稀奇的，只不过后面连的运算符刚好是括号而已。

这里我们可以通过 `lambda . operator()` 来访问这个成员，就可以看出，`operator()` 就和一个普通成员名字一样，没有区别，一样可以通过 `.` 访问。

例如，对于运算符 `+` 来说，当编译器检测到 `lambda + 2` 这样的表达式时，会自动翻译成 `lambda.operator+ (2)`，这就是所谓的运算符重载。

```cpp
struct Lambda {
    int operator+ (int a) const {
        return a + 1;
    }
};
Lambda lambda;
int ret = lambda + 2;
// 会被编译器翻译成：
int ret = lambda.operator+ (2);
```

同样的，对于 `()` 运算符，也会被编译器翻译成 `operator()` 这个函数的调用，由于对 `operator()` 函数本身的调用也需要一个括号（参数列表），所以看起来就有两个括号了。实际上根本不搭界，一个是函数名标识符的一部分，一个是产生函数调用。

```cpp
struct Lambda {
    int operator() (int a) const {
        return a + 1;
    }
};
Lambda lambda;
int ret = lambda(2);
// 会被编译器翻译成：
int ret = lambda.operator() (2);
```

这时候，去掉 `(2)` 里的参数 `2`，就变成了让你很困惑的双括号。而很多人喜欢紧挨者连写，看起来就很迷惑。

实际上，第一个 `()` 是函数名字的一部分，和 `operator` 是连在一起的，不可分割，中间也不能有其他参数。第二个 `()` 是函数参数列表，只不过这里刚好是没有参数，所以看起来也是个空括号，很多初学者看到就迷糊了，还看不懂建议从上面有一个参数的 `operator() (int a)` 看。

```cpp
struct Lambda {
    int operator() () const {
        return 1;
    }
};
Lambda lambda;
int ret = lambda();
// 会被编译器翻译成：
int ret = lambda.operator() ();
```

所以，这就是为什么说定义了 `operator()` 成员函数的类，是“函数对象”或者说“仿函数”，因为当你使用函数的语法 `lambda(2)` 调用他们时，会触发他们的成员函数 `operator()(2)` 从而用法和普通函数一样，但其实际又是对象，也就得名“函数对象”和“仿函数”了。

我建议你自己去 https://cppinsights.io 这个解构语法糖的工具网站动动手试试看：

```cpp
auto lambda = [] (int a) {
    return a + 1;
};
int ret = lambda(2);
```

实际被编译器翻译成：

```cpp
struct Lambda {
    int operator() (int a) const {
        return a + 1;
    }
};
Lambda lambda;
int ret = lambda.operator() (2);
```

而捕获了变量的：

```cpp
int x = 4;
auto lambda = [&x] (int a) {
    return a + x;
};
int ret = lambda(2);
```

实际被编译器翻译成：

```cpp
struct Lambda {
    int &x;

    Lambda(int &x_) : x(x_) {}

    int operator() (int a) const {
        return a + x;
    }
};
int x = 4;
Lambda lambda(x);
int ret = lambda.operator() (2);
```

#### 闭包捕获变量的生命周期问题

正因如此，闭包按值捕获（`[=]`）的变量，其生命周期和 Lambda 对象相同。

当 Lambda 对象被拷贝时，其按值捕获的所有变量也会被重新拷贝一份。

当 Lambda 对象被移动时，其按值捕获的所有变量也会随之一起移动。

```cpp
struct C {
    C() { fmt::println("C 默认构造"); }
    C(C const &) { fmt::println("C 拷贝构造"); }
    C(C &&) { fmt::println("C 移动构造"); }
    C &operator=(C const &) { fmt::println("C 拷贝赋值"); }
    C &operator=(C &&) { fmt::println("C 移动赋值"); }
    ~C() { fmt::println("C 析构"); }
};

C c;
fmt::println("构造 lambda");
auto lambda = [c] {};
fmt::println("拷贝 lambda 到 lambda2");
auto lambda2 = lambda;
fmt::println("移动 lambda 到 lambda3");
auto lambda3 = lambda;
```

输出：

```
C 默认构造
构造 lambda
C 拷贝构造
拷贝 lambda 到 lambda2
C 拷贝构造
移动 lambda 到 lambda3
C 移动构造
C 析构
C 析构
C 析构
C 析构
```

如果按值捕获了不能拷贝的对象（比如 `std::unique_ptr`），那么 Lambda 对象也会无法拷贝，只能移动。

```cpp
std::unique_ptr<int> p = std::make_unique<int>(10);
auto lambda = [p] {};                // 编译错误💣因为这里等价于 [p' = p]，是对 p' 的拷贝构造
auto lambda = [p = std::move(p)] {}; // 编译通过✅unique_ptr 支持移动构造
auto lambda2 = lambda;               // 编译错误💣std::unique_ptr 只支持移动，不支持拷贝
auto lambda2 = std::move(lambda);    // 编译通过✅
```

用我们之前的方法解构语法糖后：

```cpp
struct Lambda {
    std::unique_ptr<int> p;
    Lambda(std::unique_ptr<int> ptr) : p(std::move(ptr)) {}

    // Lambda(Lambda const &) = delete;  // 因为有 unique_ptr 成员，导致 Lambda 的拷贝构造函数被隐式删除

    void operator()() const {
    }
};

int main() {
    std::unique_ptr<int> p = std::make_unique<int>(10);
    Lambda lambda(p);            // 编译错误💣
    Lambda lambda(std::move(p)); // 编译通过✅
    return 0;
}
```

#### `mutable` 的函数对象

```cpp
int x = 10;
auto lambda = [x] () {
    return x++; // 编译错误💣lambda 捕获的 x 默认是只读的
};
int ret = lambda();
```

会被编译器翻译成：

```cpp
struct Lambda {
    int x;

    int operator() () const {
        return x++; // 编译错误💣const 成员函数不能修改成员变量
    }
};
int x = 10;
Lambda lambda{x};
int ret = lambda.operator() ();
```

注意到，这里的 `operator()` 成员函数有一个 `const` 修饰，意味着该成员函数不能修改其体内的变量。

所有 lambda 函数对象生成时默认，就会给他的 `operator()` 成员函数加上 `const` 修饰。

也就是说闭包捕获的变量默认是只读的，如果需要修改捕获的变量，可以给 lambda 加上 `mutable` 修饰，就加在 `()` 后面。

```cpp
int x = 10;
auto lambda = [x] () mutable {
    return x++; // 编译通过✅
};
fmt::println("lambda() = {}", lambda()); // 10
fmt::println("lambda() = {}", lambda()); // 11
fmt::println("lambda() = {}", lambda()); // 12
```

编译器翻译产生的 `Lambda` 类的成员函数，就不会带 `const` 修饰了，从而允许我们的函数体修改捕获的非引用变量。

```cpp
struct Lambda {
    int x;

    int operator() () {
        return x++; // 编译通过✅
    }
};
int x = 10;
Lambda lambda{x};
fmt::println("lambda() = {}", lambda.operator() ()); // 10
fmt::println("lambda() = {}", lambda.operator() ()); // 11
fmt::println("lambda() = {}", lambda.operator() ()); // 12
```

注意：由于使用了值捕获，lambda 修改的是在他创建时对 `x` 的一份拷贝，外面的 `x` 不会改变！

```cpp
int x = 10;
Lambda lambda{x};
fmt::println("lambda() = {}", lambda.operator() ()); // 10
fmt::println("lambda() = {}", lambda.operator() ()); // 11
fmt::println("lambda() = {}", lambda.operator() ()); // 12
fmt::println("x = {}", x);                           // 10
fmt::println("lambda.x = {}", lambda.x);             // 13
```

```cpp
int x = 10;
auto lambda = [x] () mutable {
    return x++; // 编译通过✅
};
fmt::println("ret = {}", lambda()); // 10
fmt::println("ret = {}", lambda()); // 11
fmt::println("ret = {}", lambda()); // 12
fmt::println("x = {}", x);          // 10
fmt::println("lambda.x = {}", lambda.x); // 编译错误💣编译器产生的匿名 lambda 对象中捕获产生的 x 成员变量是匿名的，无法访问
```

## 深入认识 lambda 语法

### 捕获列表语法

一个变量的三种捕获方式：

- 按值拷贝捕获 `[x]`
- 按引用捕获 `[&x]`
- 按值移动捕获 `[x = std::move(x)]`
- 按自定义表达式捕获 `[x = ...]`

批量捕获：

- 按值拷贝捕获所有用到的变量 `[=]`
- 按引用捕获所有用到的变量 `[&]`
- 多个捕获 + 默认捕获方式 `[x, y, &]` 或 `[&x, &y, =]`

#### 按值拷贝捕获

语法：`[变量名]`

按值拷贝捕获的变量，在 lambda 对象创建时，会拷贝一份捕获的变量。

lambda 捕获的变量 x 与原先 main 函数中的 x 已经是两个不同的变量，对 main 函数中 x 的修改不会影响 lambda 捕获 x 的值。

main 中的修改对 lambda 不可见。

```cpp
int main() {
    int x = 985;
    auto lambda = [x] (int i) {
        fmt::println("in lambda: x = {}", x);
    };
    fmt::println("in main: x = {}", x);
    lambda();
    x = 211;
    fmt::println("in main: x = {}", x);
    lambda();
}
```

输出：

```
in main: x = 985
in lambda: x = 985
in main: x = 211
in lambda: x = 985
```

演示：lambda 中的修改对 main 不可见。

```cpp
int main() {
    int x = 985;
    auto lambda = [x] (int i) mutable {
        fmt::println("in lambda: x = {}", x);
        x = 211;
    };
    fmt::println("in main: x = {}", x);
    lambda();
    fmt::println("in main: x = {}", x);
    lambda();
}
```

> {{ icon.tip }} 由于 lambda 按值捕获的成员默认都是不可修改（`const`），需要 `mutable` 才能修改按值捕获的成员。而按引用捕获就不需要 `mutable`，因为虽然 lambda 本身不可修改，但他指向的东西可以修改呀！

输出：

```
in main: x = 985
in lambda: x = 985
in main: x = 985
in lambda: x = 211
```

演示：main 中 x 生命周期结束后，lambda 中的 x 依然有效。

```cpp
int main() {
    std::function<void(int)> lambda;
    {
        int x = 985;
        lambda = [x] (int i) {
            fmt::println("in lambda: x = {}", x);
        };
        fmt::println("in main: x = {}", x);
        lambda();
    }
    fmt::println("in main: x 已经析构");
    lambda();
}
```

输出：

```
in main: x = 985
in lambda: x = 985
in main: x 已经析构
in lambda: x = 985
```

#### 按引用捕获

语法：`[&变量名]`

按引用捕获的变量，在 lambda 对象创建时，会创建一份指向变量的引用。

lambda 捕获的变量引用 &x 与原先 main 函数中的 x 是同一个变量，对 main 函数中 x 的修改会直接影响 lambda 捕获中 x 的值，反之亦然。

演示：main 中的修改对 lambda 可见。

```cpp
int main() {
    int x = 985;
    auto lambda = [&x] (int i) {
        fmt::println("in lambda: x = {}", x);
    };
    fmt::println("in main: x = {}", x);
    lambda();
    x = 211;
    fmt::println("in main: x = {}", x);
    lambda();
}
```

输出：

```
in main: x = 985
in lambda: x = 985
in main: x = 211
in lambda: x = 211
```

演示：lambda 中的修改对 main 也可见。

```cpp
int main() {
    int x = 985;
    auto lambda = [&x] (int i) {
        fmt::println("in lambda: x = {}", x);
        x = 211;
    };
    fmt::println("in main: x = {}", x);
    lambda();
    fmt::println("in main: x = {}", x);
    lambda();
}
```

输出：

```
in main: x = 985
in lambda: x = 985
in main: x = 211
in lambda: x = 211
```

演示：main 中 x 生命周期结束后，lambda 中的 x 将成为危险的“空悬引用（dangling-reference）”！此时再尝试访问 x，将产生未定义行为。

```cpp
int main() {
    std::function<void(int)> lambda;
    {
        int x = 985;
        lambda = [&x] (int i) {
            fmt::println("in lambda: x = {}", x);
        };
        fmt::println("in main: x = {}", x);
        lambda();
    }
    fmt::println("in main: x 已经析构");
    lambda();
}
```

输出：

```
in main: x = 985
in lambda: x = 985
in main: x 已经析构
in lambda: x = -858993460
```

> {{ icon.tip }} `-858993460` 为内存中的垃圾值，你读到的结果可能随平台，编译器版本，优化选项的不同而不同，正常读到 `985` 也是有可能的，开发者不能依赖此类随机性的结果。

> {{ icon.fun }} 正常读到 985（大学）也是有可能的。

> {{ icon.detail }} `-858993460` 是在 Windows 平台的调试模式下可能的输出，因为 Windows 倾向于把栈内存填满 `0xcccccccc` 以方便调试，其中 `0xcc` 刚好也是 `int3` 这条 x86 调试指令的二进制码，可能是为了避免指令指针执行到堆栈里去。

#### 按值移动捕获

TODO

#### 自定义表达式捕获

TODO

### lambda 中的 `auto` 类型推导

#### `auto` 推导返回类型

lambda 函数可以通过在参数列表后使用 `->` 指定函数返回类型：

```cpp
auto lambda = [] (int a) -> int {
    return a;
};
int i = lambda();
```

如果返回类型省略不写，默认是 `-> auto`，也就是根据你的 return 语句自动推导返回类型。

```cpp
auto lambda = [] (int a) {
    return a;
};
// 等价于：
auto lambda = [] (int a) -> auto {
    return a;
};
```

和普通函数返回类型声明为 `auto` 一样，会自动根据表达式为你推导返回类型：

```cpp
auto lambda = [] (int a) {
    return a; // 此表达式类型为 int
};
// 等价于：
auto lambda = [] (int a) -> int { // 所以 auto 推导出的返回类型也是 int
    return a;
};
```

```cpp
auto lambda2 = [] (int a) {
    return a * 2.0; // 此返回表达式的类型为 double
};
// 等价于：
auto lambda2 = [] (int a) -> double { // 所以 auto 推导出的返回类型也是 double
    return a * 2.0;
};
```

如果没有返回语句，那么会推导为返回 `void` 类型的 lambda。

```cpp
auto lambda = [] (int a) {
    fmt::println("a = {}", a);
};
// 等价于：
auto lambda = [] (int a) -> void {
    fmt::println("a = {}", a);
};

auto lambda = [] (int a) {
    return;
};
// 等价于：
auto lambda = [] (int a) -> void {
    return;
};
```

和函数的 `auto` 返回类型推导一样，当返回类型为 `auto` 的 lambda 具有多个返回语句时，必须保证所有分支上的返回值具有相同的类型，否则编译器报错，需要手动写出返回类型，或者把所有分支的返回值改成相同的。

```cpp
auto lambda_error = [] (double x) { // 编译错误：两个分支的返回类型不同，无法自动推导
    if (x > 0) {
        return x; // double
    } else {
        return 0; // int
    }
};

auto lambda_ok = [] (double x) { // 编译通过
    if (x > 0) {
        return x;          // double
    } else {
        return (double)0; // double
    }
};

auto lambda_also_ok = [] (double x) -> double { // 手动明确返回类型，编译也能通过
    if (x > 0) {
        return x; // double
    } else {
        return 0; // int，但会隐式转换为 double
    }
};
```

#### `auto` 推导参数类型

TODO

#### `auto` 参数实现多次实例化的应用

#### `auto &` 与 `auto const &` 的应用

#### `auto &&` 万能引用

#### `decltype(auto)` 保留真正的原始返回类型

## lambda 常见的三大用法

### 储存一个函数对象做局部变量

我们总是用 `auto` 来保存一个函数对象作为局部变量，这会自动推导 lambda 的匿名类型。

为什么不能显式写出类型名字？因为 lambda 的类型是匿名的，你无法写出类型名，只能通过 `auto` 推导。

```cpp
int b = 2;
auto lambda = [b] (int a) {
    return a + b;
};
```

> {{ icon.fun }} 这也是为什么 C++11 同时引入 `auto` 和 lambda 语法的原因。

如果你实在需要显式的类名，那就需要使用 `std::function` 容器。虽然 lambda 表达式产生的类型是匿名的，但是该类型符合“可调用”的约束，可以被 `std::function` 容器接纳。

> {{ icon.tip }} 即 lambda 类型可隐式转换为相应参数列表的 `std::function` 容器。因为 `std::function<Ret(Args)>` 容器可以接纳任何“可接受 `(Args...)` 参数调用并返回 `Ret` 类型”的任意函数对象。

```cpp
int b = 2;
std::function<int(int)> lambda = [b] (int a) {
    return a + b;
};
```

例如当我们需要把 lambda 对象推入 `vector` 等容器中时，就需要显式写出函数对象的类型，此时万能函数对象容器 `std::function` 就能派上用场了：

```cpp
// vector<auto> lambda_list;             // 错误：不支持的语法
vector<function<int(int)>> lambda_list; // OK

int b = 2;
lambda_list.push_back([b] (int a) {
    return a + b;
});
lambda_list.push_back([b] (int a) {
    return a * b;
});

for (auto lambda: lambda_list) {
    int ret = lambda(2);
    fmt::println("{}", ret);
}
```

#### 应用案例

##### 代码复用

TODO

##### 就地调用的 lambda-idiom

TODO

#### 注意捕获变量的生命周期

新手用 lambda 常见的错误就是搞不清捕获变量的生命周期，总是想当然地无脑用 `[&]`，非常危险。

如果你有“自知之明”，自知不熟悉生命周期分析，那就全部 `[=]`。

> {{ icon.tip }} 等我们稍后的 [生命周期专题课程](cpp_lifetime.md) 中介绍。

实际上，`[=]` 应该是你默认的捕获方式。

只有当类型无法拷贝会深拷贝成本过高时，才会选择性地把一些可以改成引用捕获的部分 lambda，使用 `[&]` 来捕获部分需要避免拷贝的变量，或者使用 `shared_ptr` 配合 `[=]` 将深拷贝化为浅拷贝。

> {{ icon.fun }} 一些习惯了 Python、JS 等全员 `shared_ptr` 的垃圾回收语言巨婴，一上来就全部无脑 `[&]`，用实际行动证明了智商和勇气成反比定律。

好消息是，对于代码复用和就地调用的情况，lambda 对象的生命都不会出函数体，可以安全地改成按引用捕获 `[&]`。

但是对于下面两种情况（作为参数传入和作为返回值），就不一定有这么幸运了。

总之，无论如何要保证 lambda 对象的生命周期 小于等于 按引用捕获的所有变量的生命周期。如果做不到，那就得把这些可能超出的变量改成按值捕获 `[=]`。

### 返回一个函数对象做返回值

如果你想让返回一个函数对象，分为两种情况：

就地定义（声明与定义合体）的函数，建议填写 `auto` 为返回值类型，自动推导 lambda 的匿名类型（因为你无法写出具体类型名）。

然后，在 `return` 语句中就地写出 lambda 表达式即可：

```cpp
auto make_adder(int x) {
    return [x] (int y) {
        return x + y;
    };
}
```

分离声明与定义的函数，无法使用 `auto` 推导返回类型，不得不使用万能的函数容器 `std::function` 来擦屁股：

```cpp
// adder.h
std::function<int()> make_adder(int x);

// adder.cpp
std::function<int()> make_adder(int x) {
    return [x] (int y) {
        return x + y;
    };
}
```

“函数返回一个函数对象”，这种用法在函数式编程非常常见。

#### 应用案例

例如上述的 `make_adder` 等于绑定了一个固定参数 `x` 的加法函数，之后每次调用这个返回的函数对象，就固定增加之前在 `make_adder` 参数中 `x` 的增量了。

TODO

#### 注意捕获变量的生命周期

此类“返回一个函数对象”的写法，其 lambda 捕获必须是按值捕获的！

否则，因为调用者调用返回的函数对象时，局部变量和实参所对应的函数局部栈空间已经释放，相当于在 lambda 体内存有空悬引用，导致出现未定义行为（要么直接崩溃，要么非常隐蔽地留下内存非法访问的隐患）。

```cpp
auto make_adder(int x) {
    return [x] (int y) {
        return x + y;
    };
}

int main() { // 我是调用者
    auto adder = make_adder(2);
    adder(3);  // 2 + 3 = 5
}
```

### 接受一个函数对象做参数

TODO：代码

#### 应用案例

TODO：策略模式

TODO：延迟回调

#### 注意捕获变量的生命周期

函数对象做参数的生命周期问题，需要分就地调用和延迟调用两种情况讨论。

### 生命周期问题总结：何时使用 `[=]` 或 `[&]`

如果你的智力暂不足以搞懂生命周期分析，没关系，始终使用 `[=]` 肯定没错。

> {{ icon.tip }} 一个同学询问：我口渴！在不知道他的耐受度的情况下，我肯定是直接给他吃水，而不是给他吃酒精。虽然一些孝子曰“适量”“适度”“计量”各种一连串附加条件下，宣称“酒精也是安全的”。但是“水永远是安全的”，“永远”，那我直接给他喝水，是肯定不会错的。等你长大成年了，有辨别能力了，再去根据自己的小计机瘙痒程度，选择性地喝有机溶剂。此处 `[=]` 就是这个万能的水，虽然不一定高效，但是肯定没错。初学者总是从 `[=]` 用起，等学明白了，再来尝试突破“小计机性能焦虑优化”也不迟。

如果你自认为能分得清：

- 在当前函数体内创建，当前函数体内立即调用，可以引用捕获 `[&]`，但值捕获 `[=]` 也没错。
- 返回一个 lambda，必须值捕获 `[=]`。
- 接受一个 lambda 做参数，需要进一步分为两种情况：
  - 在当前函数体内立即调用，可以引用捕获 `[&]`，但值捕获 `[=]` 也没错。
  - 作为回调函数，延迟调用，那就必须值捕获 `[=]`。

以上四种情况，分别代码演示：

```cpp
void func() {
    int i = 1;
    auto lambda = [&] () { return i; };
    lambda();
}

int main() {
    func();
}
```

```cpp
auto func() {
    int i = 1;
    return [=] () { return i; };
}

int main() {
    auto lambda = func();
    lambda();
}
```

```cpp
auto func(auto lambda) {
    lambda();
}

int main() {
    int i = 1;
    func([&] () { return i; });
}
```

```cpp
vector<function<int()>> g_callbacks;
auto func(auto lambda) {
    g_callbacks.push_back(lambda);
}

void init() {
    int i = 1;
    func([=] () { return i; });
}

int main() {
    init();
    for (auto cb: g_callbacks) {
        cb();
    }
}
```

## lambda 用于 STL 模板的仿函数参数

分为两种情况：

### 模板函数

模板函数比较简单，直接往函数参数中传入 lambda 对象即可。

`sort`：

```cpp
std::vector<int, int> a = {1, 4, 2, 8, 5, 7};
auto comp = [] (int i, int j) {
    return i < j;
};
std::sort(a.begin(), a.end(), comp);
fmt::println("a = {}", a);
```

效果：将 a 数组从大到小排序后打印。

`shared_ptr`：

```cpp
auto deleter = [] (FILE *fp) {
    fclose(fp);
};
std::shared_ptr<FILE> p(fopen("hello.txt", "r"), deleter);
```

效果：当 p 的引用计数归零时，调用 `fclose(p.get())`。

### 模板类

而模板类则需要先在模板参数中指定类型，然后在构造函数中传入参数。

```cpp
std::vector<int, int> a = {1, 4, 2, 8, 5, 7};
auto comp = [] (int i, int j) {
    return i < j;
};
std::set<int, decltype(comp)> sorted(comp);
sorted.assign(a.begin(), a.end());
a.assign(sorted.begin(), sorted.end());
fmt::println("a = {}", a);
```

效果：利用 `set` 容器有序的特点，将 a 数组从大到小排序后打印。

`unique_ptr`：

```cpp
auto deleter = [] (FILE *fp) {
    fclose(fp);
};
std::unique_ptr<FILE, decltype(deleter)> p(fopen("hello.txt", "r"), deleter);
```

效果：当 p 析构时，调用 `fclose(p.get())`。

### lambda 在 STL 中的使用案例

```cpp
TODO: count_if, erase_if, argsort
```

### 标准库自带的运算符仿函数

二元运算符

| 运算符      | 仿函数类型                         |
| ----------- | ---------------------------------- |
| `a < b`   | `std::less`                      |
| `a > b`   | `std::greater`                   |
| `a <= b`  | `std::less_equal`                |
| `a >= b`  | `std::greater_equal`             |
| `a == b`  | `std::equal_to`                  |
| `a != b`  | `std::not_equal_to`              |
| `a <=> b` | `std::compare_three_way` (C++20) |
| `a && b`  | `std::logical_and`               |
| `a \|\| b`  | `std::logical_or`                |
| `a & b`   | `std::bit_and`                   |
| `a \| b`   | `std::bit_or`                    |
| `a ^ b`   | `std::bit_xor`                   |
| `a + b`   | `std::plus`                      |
| `a - b`   | `std::minus`                     |
| `a * b`   | `std::multiplies`                |
| `a / b`   | `std::divides`                   |
| `a % b`   | `std::modulus`                   |

一元运算符

| 运算符 | 仿函数类型           |
| ------ | -------------------- |
| `!a` | `std::logical_not` |
| `~a` | `std::bit_not`     |
| `-a` | `std::negate`      |
| `a`  | `std::identity`    |

## bind 为函数对象绑定参数

原始函数：

```cpp
int hello(int x, int y) {
    fmt::println("hello({}, {})", x, y);
    return x + y;
}

int main() {
    hello(2, 3);
    hello(2, 4);
    hello(2, 5);
    return 0;
}
```

绑定部分参数：

```cpp
int hello(int x, int y) {
    fmt::println("hello({}, {})", x, y);
    return x + y;
}

int main() {
    auto hello2 = std::bind(hello, 2, std::placeholders::_1);
    hello2(3);  // hello(2, 3)
    hello2(4);  // hello(2, 4)
    hello2(5);  // hello(2, 5)
    return 0;
}
```

> {{ icon.tip }} `std::placeholders::_1` 表示 `hello2` 的第一参数。

> {{ icon.tip }} std::placeholders::_1 在 bind 表达式中位于 hello 的的第二参数位置，这意味着：把 hello2 的第一参数，传递到 hello 的第二参数上去。

绑定全部参数：

```cpp
int hello(int x, int y) {
    fmt::println("hello({}, {})", x, y);
    return x + y;
}

int main() {
    auto hello23 = std::bind(hello, 2, 3);
    hello23();  // hello(2, 3)
    return 0;
}
```

绑定引用参数：

```cpp
int inc(int &x) {
    x += 1;
}

int main() {
    int x = 0;
    auto incx = std::bind(inc, std::ref(x));
    incx();
    fmt::println("x = {}", x); // x = 1
    incx();
    fmt::println("x = {}", x); // x = 2
    return 0;
}
```

> {{ icon.warn }} 如果不使用 `std::ref`，那么 `main` 里的局部变量 `x` 不会改变！因为 `std::bind` 有一个恼人的设计：默认按拷贝捕获，会把参数拷贝一份，而不是保留引用。

有趣的是，placeholder 指定的参数，却不需要 `std::ref` 才能保持引用：

```cpp
int inc(int &x, int y) {
    x += y;
}

int main() {
    int x = 0;
    auto inc1 = std::bind(inc, std::placeholders::_1, 1);
    inc1(x);  // 此处 x 是按引用传递的
    fmt::println("x = {}", x); // x = 1
    inc1(x);
    fmt::println("x = {}", x); // x = 2
    return 0;
}
```

那是因为，`std::placeholders::_1` 指定的参数会被直接完美转发给 `inc` 里的 `x`，相当于 `inc(x, 2);`。只有捕获的参数会发生拷贝，不会完美转发。

### bind 是一个失败的设计

当我们绑定出来的函数对象还需要接受参数时，就变得尤为复杂：需要使用占位符（placeholder）。

```cpp
int func(int x, int y, int z, int &w);

int w = rand();

auto bound = std::bind(func, std::placeholders::_2, 1, std::placeholders::_1, std::ref(w)); //

int res = bound(5, 6); // 等价于 func(6, 1, 5, w);
```

这是一个绑定器，把 `func` 的第二个参数和第四个参数固定下来，形成一个新的函数对象，然后只需要传入前面两个参数就可以调用原来的函数了。

这是一个非常旧的技术，C++98 时代就有了。但是，现在有了 Lambda 表达式，可以更简洁地实现：

```cpp
int func(int x, int y, int z, int &w);

int w = rand();

auto lambda = [&w](int x, int y) { return func(y, 1, x, w); };

int res = lambda(5, 6);
```

Lambda 表达式有许多优势：

- 简洁：不需要写一大堆看不懂的 `std::placeholders::_1`，直接写变量名就可以了。
- 灵活：可以在 Lambda 中使用任意多的变量，调整顺序，而不仅仅是 `std::placeholders::_1`。
- 易懂：写起来和普通函数调用一样，所有人都容易看懂。
- 捕获引用：`std::bind` 不支持捕获引用，总是拷贝参数，必须配合 `std::ref` 才能捕获到引用。而 Lambda 可以随意捕获不同类型的变量，按值（`[x]`）或按引用（`[&x]`），还可以移动捕获（`[x = move(x)]`），甚至捕获 this（`[this]`）。
- 夹带私货：可以在 lambda 体内很方便地夹带其他额外转换操作，比如：

```cpp
auto lambda = [&w](int x, int y) { return func(y + 8, 1, x * x, ++w) * 2; };
```

#### bind 的历史

为什么 C++11 有了 Lambda 表达式，还要提出 `std::bind` 呢？

虽然 bind 和 lambda 看似都是在 C++11 引入的，实际上 bind 的提出远远早于 lambda。

> {{ icon.fun }} 标准委员会：我们不生产库，我们只是 boost 的搬运工。

当时还是 C++98，由于没有 lambda，难以创建函数对象，“捕获参数”非常困难。

为了解决“捕获难”问题，在第三方库 boost 中提出了 `boost::bind`，由于当时只有 C++98，很多有益于函数式编程的特性都没有，所以实现的非常丑陋。

例如，因为 C++98 没有变长模板参数，无法实现 `<class ...Args>`。所以实际上当时 boost 所有支持多参数的函数，实际上都是通过：

```cpp
void some_func();
void some_func(int i1);
void some_func(int i1, int i2);
void some_func(int i1, int i2, int i3);
void some_func(int i1, int i2, int i3, int i4);
// ...
```

这样暴力重载几十个函数来实现的，而且参数数量有上限。通常会实现 0 到 20 个参数的重载，更多就不支持了。

例如，我们知道现在 bind 需要配合各种 `std::placeholders::_1` 使用，有没有想过这套丑陋的占位符是为什么？为什么不用 `std::placeholder<1>`，这样不是更可扩展吗？

没错，当时 `boost::bind` 就是用暴力重载几十个参数数量不等的函数，排列组合，嗯是排出来的，所以我们会看到 `boost::placeholders` 只有有限个数的占位符数量。

糟糕的是，标准库的 `std::bind` 把 `boost::bind` 原封不动搬了过来，甚至 `placeholders` 的暴力组合也没有变，造成了 `std::bind` 如今丑陋的接口。

人家 `boost::bind` 是因为不能修改语言语法，才只能那样憋屈的啊？可现在你码是标准委员会啊，你可以修改语言语法啊？

然而，C++ 标准的更新是以“提案”的方式，逐步“增量”更新进入语言标准的。即使是在 C++98 到 C++11 这段时间内，内部也是有一个很长的消化流程的，也就是说有很多子版本，只是对外看起来好像只有一个 C++11。

比方说，我 2001 年提出 `std::bind` 提案，2005 年被批准进入未来将要发布的 C++11 标准。然后又一个人在 2006 年提出其实不需要 bind，完全可以用更好的 lambda 语法来代替 bind，然后等到了 2008 年才批准进入即将发布的 C++11 标准。但是已经进入标准的东西就不会再退出了，哪怕还没有发布。就这样 bind 和 lambda 同时进入了标准。

所以闹了半天，lambda 实际上是 bind 的上位替代，有了 lambda 根本不需要 bind 的。只不过是由于 C++ 委员会前后扯皮的“制度优势”，导致 bind 和他的上位替代 lambda 同时进入了 C++11 标准一起发布。

> {{ icon.fun }} 这下看懂了。

很多同学就不理解，小彭老师说“lambda 是 bind 的上位替代”，他就质疑“可他们不都是 C++11 提出的吗？”

有没有一种可能，C++11 和 C++98 之间为什么年代差了那么久远，就是因为一个标准一拖再拖，内部实际上已经迭代了好几个小版本了，才发布出来。

> {{ icon.story }} 再举个例子，CTAD 和 `optional` 都是 C++17 引入的，为什么还要 `make_optional` 这个帮手函数？不是说 CTAD 是 `make_xxx` 的上位替代吗？可见，C++ 标准中这种“同一个版本内”自己打自己耳光的现象比比皆是。

> {{ icon.fun }} 所以，现在还坚持用 bind 的，都是些 2005 年前后在象牙塔接受 C++ 教育，但又不肯“终身学习”的劳保。这批劳保又去“上岸”当“教师”，继续复制 2005 年的错误毒害青少年，实现了劳保的再生产。

#### thread 膝盖中箭

糟糕的是，bind 的这种荼毒，甚至影响到了线程库：`std::thread` 的构造函数就是基于 `std::bind` 的！

这导致了 `std::thread` 和 `std::bind` 一样，无法捕获引用。

```cpp
void thread_func(int &x) {
    x = 42;
}

int x = 0;
std::thread t(thread_func, x);
t.join();
printf("%d\n", x); // 0
```

为了避免踩到 bind 的坑，我建议所有同学，构造 `std::thread` 时，统一只指定“单个参数”，也就是函数本身。如果需要捕获参数，请使用 lambda。因为 lambda 中，捕获了哪些变量，参数的顺序是什么，哪些捕获是引用，哪些捕获是拷贝，非常清晰。

```cpp
void thread_func(int &x) {
    x = 42;
}

int x = 0;
std::thread t([&x] {  // [&x] 表示按引用捕获 x；如果写作 [x]，那就是拷贝捕获
    thread_func(x);
});
t.join();
printf("%d\n", x); // 42
```

#### 案例：绑定随机数生成器

bind 写法：

```cpp
std::mt19937 gen(seed);
std::uniform_real_distribution<double> uni(0, 1);
auto frand = std::bind(uni, std::ref(gen));
double x = frand();
double y = frand();
```

改用 lambda：

```cpp
std::mt19937 gen(seed);
std::uniform_real_distribution<double> uni(0, 1);
auto frand = [uni, &gen] {
    return uni(gen);
};
double x = frand();
double y = frand();
```

### `std::bind_front` 和 `std::bind_back`

C++17 引入了两个新绑定函数：

- `std::bind_front`：绑定最前的若干个参数，后面的参数自动添加占位符；
- `std::bind_back`：绑定末尾的若干个参数，前面的参数自动添加占位符。

和普通的 `std::bind` 相比有什么好处呢？

对于函数参数非常多，但实际只需要绑定一两个参数的情况，用 `std::bind` 会需要添加非常多的 placeholder，数量和函数的剩余参数数量一样多。而 `std::bind_front` 则相当于一个简写，后面的占位符可以省略不写了。

例如绑定 x = 42：

```cpp
int func(int x, int y, int z);

auto bound = std::bind(func, 42, std::placeholders::_1, std::placeholders::_2);
// 等价于：
auto bound = std::bind_front(func, 42);
```

绑定 z = 42：

```cpp
int func(int x, int y, int z);

auto bound = std::bind(func, std::placeholders::_1, std::placeholders::_2, 42);
// 等价于：
auto bound = std::bind_back(func, 42);
```

可以看到，使用这两个新绑定函数明显写的代码少了。

> {{ icon.tip }} 其中最常用的是 `std::bind_front`，用于绑定类成员的 `this` 指针。

### 案例：绑定成员函数

使用“成员函数指针”语法（这一奇葩语法在 C++98 就有）配合 `std::bind`，可以实现绑定一个类型的成员函数：

```cpp
struct Class {
    void world() {
        puts("world!");
    }

    void hello() {
        auto memfn = std::bind(&Class::world, this); // 将 this->world 绑定成一个可以延后调用的函数对象
        memfn();
        memfn();
    }
}
```

不就是捕获 this 吗？我们 lambda 也可以轻易做到！且无需繁琐地写出 this 类的完整类名，还写个脑瘫 `&::` 强碱你的键盘。

```cpp
struct Class {
    void world() {
        puts("world!");
    }

    void hello() {
        auto memfn = [this] {
            world(); // 等价于 this->world()
        };
        memfn();
        memfn();
    }
}
```

bind 的缺点是，当我们的成员函数含有多个参数时，bind 就非常麻烦了：需要一个个写出 placeholder，而且数量必须和 `world` 的参数数量一致。每次 `world` 要新增参数时，所有 bind 的地方都需要加一下 placeholder，非常沙雕。

```cpp
struct Class {
    void world(int x, int y) {
        printf("world(%d, %d)\n");
    }

    void hello() {
        auto memfn = std::bind(&Class::world, this, std::placeholders::_1, std::placeholders::_2);
        memfn(1, 2);
        memfn(3, 4);
    }
}
```

而且，如果有要绑定的目标函数有多个参数数量不同的重载，那 bind 就完全不能工作了！

```cpp
struct Class {
    void world(int x, int y) {
        printf("world(%d, %d)\n");
    }

    void world(double x) {
        printf("world(%d)\n");
    }

    void hello() {
        auto memfn = std::bind(&Class::world, this, std::placeholders::_1, std::placeholders::_2);
        memfn(1, 2);
        memfn(3.14); // 编译出错！死扣占位符的 bind 必须要求两个参数，即使 world 明明有单参数的重载

        auto memfn_1arg = std::bind(&Class::world, this, std::placeholders::_1);
        memfn_1arg(3.14); // 必须重新绑定一个“单参数版”才 OK
    }
}
```

## 使用 `std::bind_front` 代替

为了解决 bind 不能捕获多参数重载的情况，C++17 引入了 `std::bind_front` 和 `std::bind_back`，他们不需要 placeholder，但只能用于要绑定的参数在最前或者最后的特殊情况。

其中 `std::bind_front` 对于我们只需要把第一个参数绑定为 `this`，其他参数如数转发的场景，简直是雪中送炭！

```cpp
struct Class {
    void world(int x, int y) {
        printf("world(%d, %d)\n");
    }

    void world(double x) {
        printf("world(%d)\n");
    }

    void hello() {
        auto memfn = std::bind_front(&Class::world, this);
        memfn(1, 2);
        memfn(3.14); // OK！
    }
}
```

```cpp
auto memfn = std::bind_front(&Class::world, this); // C++17 的 bind 孝子补救措施
auto memfn = BIND(world, this);                    // 小彭老师的 BIND 宏，C++14 起可用
```

你更喜欢哪一种呢？

#### 使用 lambda 代替

而 C++14 起 lambda 支持了变长参数，就不用这么死板：

```cpp
struct Class {
    void world(int x, int y) {
        printf("world(%d, %d)\n");
    }

    void world(double x) {
        printf("world(%d)\n");
    }

    void hello() {
        auto memfn = [this] (auto ...args) { // 让 lambda 接受任意参数
            world(args...); // 拷贝转发所有参数给 world
        };
        memfn(1, 2); // 双参数：OK
        memfn(3.14); // 单参数：OK
    }
}
```

更好的是配合 `forward` 实现参数的完美转发：

```cpp
struct Class {
    void world(int &x, int &&y) {
        printf("world(%d, %d)\n");
        ++x;
    }

    void world(double const &x) {
        printf("world(%d)\n");
    }

    void hello() {
        auto memfn = [this] (auto &&...args) { // 让 lambda 接受万能引用做参数
            world(std::forward<decltype(args)>(args)...); // 通过 FWD 完美转发给 world，避免引用退化
        };
        int x = 1;
        memfn(x, 2); // 双参数：OK
        memfn(3.14); // 单参数：OK
    }
}
```

### bind 与标准库自带的运算符仿函数配合

TODO：`std::less` 和 `std::bind`

### 函数指针是 C 语言陋习，改掉

## lambda 进阶案例

### lambda 实现递归

### lambda 避免全局重载函数捕获为变量时恼人的错误

### lambda 配合 if-constexpr 实现编译期三目运算符

### 推荐用 C++23 的 `std::move_only_function` 取代 `std::function`

通过按值移动捕获 `[p = std::move(p)]`，lambda 可以持有一个 unique_ptr 作为捕获变量。

但是，我们会发现，这样创建出来的 lambda，存入 `std::function` 时会报错：

TODO: 代码

### 无状态 lambda 隐式转换为函数指针

### 与 `std::variant` 和 `std::visit` 配合实现动态多态

TODO: 代码案例

在之后的 [`std::variant` 专题章节](design_variant.md)中会进一步介绍。

### 配合 `shared_from_this` 实现延长 this 生命周期

### `mutable` lambda 实现计数器

### C++20 中的 lambda 扩展用法
