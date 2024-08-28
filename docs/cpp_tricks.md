# 应知应会 C++ 小技巧

[TOC]

## 交换两个变量

```cpp
int a = 42;
int b = 58;
```

现在你想交换这两个变量。

```cpp
int tmp = a;
a = b;
b = tmp;
```

但是标准库提供了更好的方法：

```cpp
std::swap(a, b);
```

这个方法可以交换任意两个同类型的值，包括结构体、数组、容器等。

> {{ icon.tip }} 只需要 `#include <utility>` 就可以使用！

## RAII 地分配一段内存空间

小彭老师：不要让我看到 new 和 delete。

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

## 读取整个文件到字符串

```cpp
TODO
```

## 位域（bit-field）

```cpp
TODO
```

## 别再写构造函数啦！

```cpp
// C++98
struct Student {
    string name;
    int age;
    int id;

    Student(string name_, int age_, int id_) : name(name_), age(age_), id(id_) {}
};

Student stu("侯捷老师", 42, 123);
```

C++98 需要手动书写构造函数，非常麻烦！而且几乎都是重复的。

C++11 中，平凡的结构体类型不需要再写构造函数了，只需用 `{}` 就能对成员依次初始化：

```cpp
// C++11
struct Student {
    string name;
    int age;
    int id;
};

Student stu{"小彭老师", 24, 123};
```

这被称为**聚合初始化** (aggergate initialize)。只要你的类没有自定义构造函数，没有 private 成员，都可以用 `{}` 聚合初始化。

好消息：C++20 中，聚合初始化也支持 `()` 了，用起来就和传统的 C++98 构造函数一样！

```cpp
// C++20
Student stu("小彭老师", 24, 123);
```

聚合初始化还可以指定默认值：

```cpp
// C++11
struct Student {
    string name;
    int age;
    int id = 9999;
};

Student stu{"小彭老师", 24};
// 等价于：
Student stu{"小彭老师", 24, 9999};
```

C++20 开始，`{}` 聚合初始化还可以根据每个成员的名字来指定值：

```cpp
Student stu{.name = "小彭老师", .age = 24, .id = 9999};
// 等价于：
Student stu{"小彭老师", 24, 9999};
```

好处是，即使不慎写错参数顺序也不用担心。

```cpp
Student stu{.name = "小彭老师", .age = 24, .id = 9999};
Student stu{.name = "小彭老师", .id = 9999, .age = 24};
```

## 别再写拷贝构造函数啦！

只有当你需要有“自定义钩子逻辑”的时候，才需要自定义构造函数。

```cpp
struct Student {
    string name;
    int age;
    int id;

    Student(string name_, int age_, int id_) : name(name_), age(age_), id(id_) {}

    Student(Student const &other) : name(other.name), age(other.age), id(other.id) {
        std::cout << "拷贝构造\n";
    }

    Student &operator=(Student const &other) {
        name = other.name;
        age = other.age;
        id = other.id;
        std::cout << "拷贝赋值\n";
        return *this;
    }
};

Student stu1("侯捷老师", 42, 123);
Student stu2 = stu1;  // 拷贝构造
stu2 = stu1;          // 拷贝赋值
```

如果你不需要这个 `std::cout`，只是平凡地拷贝所有成员，完全可以不写，让编译器自动生成拷贝构造函数、拷贝赋值函数、移动构造函数、移动赋值函数：

```cpp
struct Student {
    string name;
    int age;
    int id;

    Student(string name_, int age_, int id_) : name(name_), age(age_), id(id_) {}

    // 编译器自动生成 Student(Student const &other)
    // 编译器自动生成 Student &operator=(Student const &other)
};

Student stu1("侯捷老师", 42, 123);
Student stu2 = stu1;  // 拷贝构造
stu2 = stu1;          // 拷贝赋值
assert(stu2.name == "侯捷老师");
```

总之，很多 C++ 教材把拷贝/移动构造函数过于夸大，搞得好像每个类都需要自己定义一样。

实际上，只有在“自己实现容器”的情况下，才需要自定义拷贝构造函数。可是谁会整天手搓容器？

大多数情况下，我们只需要在类里面存 vector、string 等封装好的容器，编译器默认生成的拷贝构造函数会自动调用他们的拷贝构造函数，用户只需专注于业务逻辑即可，不需要操心底层细节。

## 提前返回

```cpp
void babysitter(Baby *baby) {
    if (!baby->is_alive()) {
        puts("宝宝已经去世了");
    } else {
        puts("正在检查宝宝喂食情况...");
        if (baby->is_feeded()) {
            puts("宝宝已经喂食过了");
        } else {
            puts("正在喂食宝宝...");
            puts("正在调教宝宝...");
            puts("正在安抚宝宝...");
        }
    }
}
```

这个函数有很多层嵌套，很不美观。用**提前返回**的写法来优化：

```cpp
void babysitter(Baby *baby) {
    if (!baby->is_alive()) {
        puts("宝宝已经去世了");
        return;
    }
    puts("正在检查宝宝喂食情况...");
    if (baby->is_feeded()) {
        puts("宝宝已经喂食过了");
        return;
    }
    puts("正在喂食宝宝...");
    puts("正在调教宝宝...");
    puts("正在安抚宝宝...");
}
```

## 立即调用的 Lambda

有时，需要在一个列表里循环查找某样东西，也可以用提前返回的写法优化：

```cpp
bool find(const vector<int> &v, int target) {
    for (int i = 0; i < v.size(); ++i) {
        if (v[i] == target) {
            return true;
        }
    }
    return false;
}
```

可以包裹一个立即调用的 Lambda 块 `[&] { ... } ()`，限制提前返回的范围：

```cpp
void find(const vector<int> &v, int target) {
    bool found = [&] {
        for (int i = 0; i < v.size(); ++i) {
            if (v[i] == target) {
                return true;
            }
        }
        return false;
    } ();
    if (found) {
        ...
    }
}
```

## Lambda 复用代码

```cpp
vector<string> spilt(string str) {
    vector<string> list;
    string last;
    for (char c: str) {
        if (c == ' ') {
            list.push_back(last);
            last.clear();
        } else {
            last.push_back(c);
        }
    }
    list.push_back(last);
    return list;
}
```

上面的代码可以用 Lambda 复用：

```cpp
vector<string> spilt(string str) {
    vector<string> list;
    string last;
    auto push = [&] {
        list.push_back(last);
        last.clear();
    };
    for (char c: str) {
        if (c == ' ') {
            push();
        } else {
            last.push_back(c);
        }
    }
    push();
    return list;
}
```

## 类内静态成员 inline

在头文件中定义结构体的 static 成员时：

```cpp
struct Class {
    static Class instance;
};
```

会报错 `undefined reference to 'Class::instance'`。这是说的你需要找个 .cpp 文件，写出 `Class Class::instance` 才能消除该错误。

C++17 中，只需加个 `inline` 就能解决！

```cpp
struct Class {
    inline static Class instance;
};
```

## 别再 `[]` 啦！

你知道吗？在 map 中使用 `[]` 查找元素，如果不存在，会自动创建一个默认值。这个特性有时很方便，但如果你不小心写错了，就会在 map 中创建一个多余的默认元素。

```cpp
map<string, int> table;
table["小彭老师"] = 24;

cout << table["侯捷老师"];
```

table 中明明没有 "侯捷老师" 这个元素，但由于 `[]` 的特性，他会默认返回一个 0，不会爆任何错误！

改用更安全的 `at()` 函数，当查询的元素不存在时，会抛出异常，方便你调试：

```cpp
map<string, int> table;
table.at("小彭老师") = 24;

cout << table.at("侯捷老师");  // 抛出异常
```

`[]` 真正的用途是“写入新元素”时，如果元素不存在，他可以自动帮你创建一个默认值，供你以引用的方式赋值进去。

检测元素是否存在可以用 `count`：

```cpp
if (table.count("小彭老师")) {
    return table.at("小彭老师");
} else {
    return 0;
}
```

即使你想要默认值 0 这一特性，`count` + `at` 也比 `[]` 更好，因为 `[]` 的默认值是会对 table 做破坏性修改的，这导致 `[]` 需要 `map` 的声明不为 `const`：

```cpp
map<string, int> table;
return table["小彭老师"]; // 如果"小彭老师"这一键不存在，会创建"小彭老师"并设为默认值 0
```

```cpp
const map<string, int> table;
return table["小彭老师"]; // 编译失败！[] 需要非 const 的 map 对象，因为他会破坏性修改
```

> {{ icon.tip }} 更多 map 知识请看我们的 [map 专题课](stl_map.md)。

## 别再 make_pair 啦！

```cpp
map<string, int> table;
table.insert(pair<string, int>("侯捷老师", 42));
```

为避免写出类型名的麻烦，很多老师都会让你写 make_pair：

```cpp
map<string, int> table;
table.insert(make_pair("侯捷老师", 42));
```

然而 C++11 提供了更好的写法，那就是通过 `{}` 隐式构造，不用写出类型名或 make_pair：

```cpp
map<string, int> table;
table.insert({"侯捷老师", 42});
```

> {{ icon.fun }} 即使你出于某种“抖m”情节，还想写出类型名，也可以用 C++17 的 CTAD 语法，免去模板参数：

```cpp
map<string, int> table;
table.insert(pair("侯捷老师", 42));

// tuple 也支持 CTAD：
auto t = tuple("侯捷老师", 42, string("小彭老师"));
// 等价于：
auto t = make_tuple("侯捷老师", 42, string("小彭老师"));

println("{}", typeid(t).name()); // tuple<const char *, int, string>
```

## insert 不会替换现有值哦

```cpp
map<string, int> table;
table.insert({"小彭老师", 24});
table.insert({"小彭老师", 42});
```

这时，`table["小彭老师"]` 仍然会是 24，而不是 42。因为 insert 不会替换 map 里已经存在的值。

如果希望如果已经存在时，替换现有元素，可以使用 `[]` 运算符：

```cpp
map<string, int> table;
table["小彭老师"] = 24;
table["小彭老师"] = 42;
```

C++17 提供了比 `[]` 运算符更适合覆盖性插入的 `insert_or_assign` 函数：

```cpp
map<string, int> table;
table.insert_or_assign("小彭老师", 24);
table.insert_or_assign("小彭老师", 42);
```

好处：`insert_or_assign` 不需要值类型支持默认构造，可以避免一次默认构造函数 + 一次移动赋值函数的开销。

> {{ icon.tip }} 建议把 `insert_or_assign` 改名成 `set`，`at` 改名成 `get`；只是由于历史原因名字迷惑了。

## 一边遍历 map，一边删除？

```cpp
map<string, int> table;
for (auto it = table.begin(); it != table.end(); ++it) {
    if (it->second < 0) {
        table.erase(it);
    }
}
```

会发生崩溃！看来 map 似乎不允许在遍历的过程中删除？不，只是你的写法有错误：

```cpp
map<string, int> table;
for (auto it = table.begin(); it != table.end(); ) {
    if (it->second < 0) {
        it = table.erase(it);
    } else {
        ++it;
    }
}
```

C++20 引入了更好的 erase_if 全局函数，不用手写上面这么麻烦的代码：

```cpp
map<string, int> table;
erase_if(table, [](pair<string, int> it) {
    return it.second < 0;
});
```

## 高效删除单个 vector 元素

```cpp
vector<int> v = {48, 23, 76, 11, 88, 63, 45, 28, 59};
```

众所周知，在 vector 中删除元素，会导致后面的所有元素向前移动，十分低效。复杂度：$O(n)$

```cpp
// 直接删除 v[3]
v.erase(v.begin() + 3);
```

如果不在乎元素的顺序，可以把要删除的元素和最后一个元素 swap，然后 pop_back。复杂度：$O(1)$

```cpp
// 把 v[3] 和 v[v.size() - 1] 位置对调
swap(v[3], v[v.size() - 1]);
// 然后删除 v[v.size() - 1]
v.pop_back();
```

这样就不用移动一大堆元素了。这被称为 back-swap-erase。

## 批量删除部分 vector 元素

vector 中只删除一个元素需要 $O(n)$。如果一边遍历，一边删除多个符合条件的元素，就需要复杂度 $O(n^2)$ 了。

标准库提供了 `remove` 和 `remove_if` 函数，其内部采用类似 back-swap-erase 的方法，先把要删除的元素移动到末尾。然后一次性 `erase` 掉末尾同样数量的元素。

且他们都能保持顺序不变。

删除所有值为 42 的元素：

```cpp
vector<int> v;
v.erase(remove(v.begin(), v.end(), 42), v.end());
```

删除所有值大于 0 的元素：

```cpp
vector<int> v;
v.erase(remove_if(v.begin(), v.end(), [](int x) {
    return x > 0;
}), v.end());
```

现在 C++20 也引入了全局函数 erase 和 erase_if，使用起来更加直观：

```cpp
vector<int> v;
erase(v, 42);       // 删除所有值为 42 的元素
erase_if(v, [](int x) {
    return x > 0;   // 删除所有值大于 0 的元素
});
```

## 保持有序的 vector

如果你想要维护一个有序的数组，用 `lower_bound` 或 `upper_bound` 来插入元素，保证插入后仍保持有序：

```cpp
vector<int> s;
s.push_back(1);
s.push_back(2);
s.push_back(4);
s.push_back(6);
// s = { 1, 2, 4, 6 }
s.insert(lower_bound(s.begin(), s.end(), 3), 3);
// s = { 1, 2, 3, 4, 6 }
s.insert(lower_bound(s.begin(), s.end(), 5), 5);
// s = { 1, 2, 3, 4, 5, 6 }
```

有序数组中，可以利用 `lower_bound` 或 `upper_bound` 快速二分查找到想要的值：

```cpp
vector<int> s;
s.push_back(1);
s.push_back(2);
s.push_back(4);
s.push_back(6);
// s = { 1, 2, 4, 6 }
lower_bound(s.begin(), s.end(), 3); // s.begin() + 2;
lower_bound(s.begin(), s.end(), 5); // s.begin() + 3;
```

有序 vector 应用案例：利用 CDF 积分 + 二分法可以实现生成任意指定分布的随机数。

例如抽卡概率要求：

- 2% 出金卡
- 10% 出蓝卡
- 80% 出白卡
- 8% 出答辩

```cpp
vector<double> probs = {0.02, 0.1, 0.8, 0.08};
vector<double> cdf;
// 计算 probs 的 CDF 积分，存入 cdf 数组
std::partial_sum(probs.begin(), probs.end(), std::back_inserter(cdf));
// cdf = {0.02, 0.12, 0.92, 1.00} 是一个有序 vector，可以运用二分法

vector<string> result = {"金卡", "蓝卡", "白卡", "答辩"};
// 生成 100 个随机数：
for (int i = 0; i < 100; ++i) {
    double r = rand() / (RAND_MAX + 1.);
    int index = lower_bound(cdf.begin(), cdf.end(), r) - cdf.begin();
    cout << "你抽到了" << result[index] << endl;
}
```

## C++ 随机数的正确生成方式

```cpp
// 错误的写法：
int r = rand() % 10; // 这样写是错误的！
```

rand() 的返回值范围是 [0, RAND_MAX]，RAND_MAX 在不同平台下不同，在 Windows 平台的是 32767，即 rand() 只能生成 0～32767 之间的随机数。

如果想要生成 0～9 之间的随机数，最简单的办法是：

```cpp
int r = rand() % 10;
```

然而这种方法有个致命的问题：不同的随机数生成概率不一样。

例如把 [0, RAND_MAX] 均匀地分成 10 份，每份 3276.7。那么 0～6 之间的数字出现的概率是 3276.7 / 32767 = 10.0003%，而 7～9 之间的数字出现的概率是 3276.7 / 32767 = 9.997%。

这样就不是真正的均匀分布，这可能会影响程序的正确性。

- 当模数大的时候不均匀性会变得特别明显，例如 `rand() % 10000`。
- RAND_MAX 在不同平台不同的特性也让跨平台开发者很头大。
- `rand` 使用全局变量存储种子，对多线程不友好。
- 无法独立的为多个生成序列设定独立的种子，一些游戏可能需要用到多个随机序列，各自有独立的种子。
- 只能生成均匀分布的整数，不能生成幂率分布、正太分布等，生成浮点数也比较麻烦。
- 使用 `srand(time(NULL))` 无法安全地生成随机数的初始种子，容易被黑客预测并攻击。
- `rand` 的算法实现没有官方规定，在不同平台各有不同，产生的随机数序列可能不同。

为此，C++ 提出了更加专业的随机数生成器：`<random>` 库。

```cpp
// 使用 <random> 库生成 0～9 之间的随机数：
#include <random>
#include <iostream>

int main() {
    uint64_t seed = std::random_device()();
    std::mt19937 gen(seed);
    std::uniform_int_distribution<int> dis(0, 9);

    for (int i = 0; i < 100; ++i) {
        int r = dis(gen);
        std::cout << r << " ";
    }
}
```

这样就可以生成 0～9 之间的均匀分布的随机数了。

- `std::random_device` 是一个随机数种子生成器，它会利用系统的随机设备（如果有的话，否则会抛出异常）生成一个安全的随机数种子，黑客无法预测。
- `std::mt19937` 是一个随机数生成器，它会利用初始种子生成一个随机数序列。并且必定是 MT19937 这个高强度的随机算法，所有平台都一样。
- `std::uniform_int_distribution` 是一个分布器，它可以把均匀分布的随机数映射到我们想要的上下界中。里面的实现类似于 `gen() % 10`，但通过数学机制保证了绝对均匀性。

类似的还有 `std::uniform_real_distribution` 用于生成浮点数，`std::normal_distribution` 用于生成正太分布的随机数，`std::poisson_distribution` 用于生成泊松分布的随机数，等等。

如果喜欢老式的函数调用风格接口，可以封装一个新的 C++ 重置版安全 `rand`：

```cpp
thread_local std::mt19937 gen(std::random_device()()); // 每线程一个，互不冲突

int randint(int min, int max) {
    return std::uniform_int_distribution<int>(min, max)(gen);
}

float randfloat(float min, float max) {
    return std::uniform_real_distribution<float>(min, max)(gen);
}
```

## const 居然应该后置...

众所周知，`const` 在指针符号 `*` 的前后，效果是不同的。

```cpp
const int *p;
int *const p;
```

你能看出来上面这个 const 分别修饰的是谁吗？

```cpp
const int *p;  // 指针指向的 int 不可变
int *const p;  // 指针本身不可改变指向
```

为了看起来更加明确，我通常都会后置所有的 const 修饰。

```cpp
int const *p;  // 指针指向的 int 不可变
int *const p;  // 指针本身不可改变指向
```

这样就一目了然，const 总是在修饰他前面的东西，而不是后面。

为什么 `int *const` 修饰的是 `int *` 也就很容易理解了。

```cpp
int const i;
int const *p;
int *const q;
int const &r;
```

举个例子：

```cpp
int i, j;
int *const p = &i;
*p = 1;  // OK：p 指向的对象可变
p = &j;  // 错误：p 本身不可变，不能改变指向
```

```cpp
int i, j;
int const *p = &i;
*p = 1;  // 错误：p 指向的对象不可变
p = &j;  // OK：p 本身可变，可以改变指向
```

```cpp
int i, j;
int const *const p = &i;
*p = 1;  // 错误：p 指向的对象不可变
p = &j;  // 错误：p 本身也不可变，不能改变指向
```

> {{ icon.tip }} `int const *` 和 `const int *` 等价！只有 `int *const` 是不同的。

## cout 不需要 endl

```cpp
int a = 42;
printf("%d\n", a);
```

万一你写错了 `%` 后面的类型，编译器不会有任何报错，留下隐患。

```cpp
int a = 42;
printf("%s\n", a);  // 编译器不报错，但是运行时会崩溃！
```

C++ 中有更安全的输出方式 `cout`，通过 C++ 的重载机制，无需手动指定 `%`，自动就能推导类型。

```cpp
int a = 42;
cout << a << endl;
double d = 3.14;
cout << d << endl;
```

```cpp
cout << "Hello, World!" << endl;
```

endl 是一个特殊的流操作符，作用等价于先输出一个 `'\n'` 然后 `flush`。

```cpp
cout << "Hello, World!" << '\n';
cout.flush();
```

但实际上，输出流 cout 默认的设置就是“行刷新缓存”，也就是说，检测到 `'\n'` 时，就会自动刷新一次，根本不需要我们手动刷新！

如果还用 endl 的话，就相当于刷新了两次，浪费性能。

所以，我们只需要输出 `'\n'` 就可以了，每次换行时 cout 都会自动刷新，endl 是一个典型的以讹传讹错误写法。

```cpp
cout << "Hello, World!" << '\n';
```

## 多线程中 cout 出现乱序？

同学：小彭老师，我在多线程环境中使用：

```cpp
cout << "the answer is " << 42 << '\n';
```

发现输出乱套了！这是不是说明 cout 不是**多线程安全**的呢？

小彭老师：cout 是一个“同步流”，是**多线程安全**的，错误的是你的使用方式。

> {{ icon.story }} 如果他不多线程安全，那多线程地调用他就不是输出乱序，而是程序崩溃了。

但是，cout 的线程安全，只能保证每一次 `operator<<` 都是原子的，每一次单独的 `operator<<` 不会被其他人打断。

但众所周知，cout 为了支持级联调用，他的 `operator<<` 都是返回自己的，上面的代码实际上等价于分别三次调用 `cout` 的 `operator<<`。

```cpp
cout << "the answer is " << 42 << '\n';
// 等价于：
cout << "the answer is ";
cout << 42;
cout << '\n';
```

变成了三次 `operator<<`，每一次都是“各自”原子的，但三个原子加在一起就不是原子了。

> {{ icon.fun }} 而是分子了 :)

他们中间可能穿插了其他线程的 cout，从而导致你 `"the answer is"` 打印完后，被其他线程的 `'\n'` 插入进来，导致换行混乱。

> {{ icon.tip }} 更多细节请看我们的 [多线程专题](threading.md)。

解决方法是，先创建一个只属于当前线程的 `ostringstream`，最后一次性调用一次 cout 的 `operator<<`，让“原子”的单位变成“一行”而不是一个字符串。

```cpp
ostringstream oss;
oss << "the answer is " << 42 << '\n';
cout << oss.str();
```

或者，使用 `std::format`：

```cpp
cout << std::format("the answer is {}\n", 42);
```

总之，就是要让 `operator<<` 只有一次。

建议各位升级到 C++23，然后改用 `std::println` 吧：

```cpp
std::println("the answer is {}", 42);
```

## 函数参数也可以 auto

大家都知道，函数的返回类型可以声明为 `auto`，让其自动推导。

```cpp
auto func() {  // int func();
    return 1;
}
```

但你知道从 C++20 开始，参数也可以声明为 auto 了吗？

```cpp
auto func(auto x) {  // T func(T x);
    return x * x;
}

func(1);    // func(int)
func(3.14); // func(double)
```

等价于以下“模板函数”的传统写法：

```cpp
template <typename T>
T func(T x) {
    return x * x;
}

func(1);    // func<int>(int)
func(3.14); // func<double>(double)
```

因为是模板函数，所以也很难分离声明和定义，只适用于头文件中就地定义函数的情况。

## bind 是历史糟粕，应该由 Lambda 表达式取代

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

### bind 的历史

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

### thread 膝盖中箭

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

### 举个绑定随机数生成器例子

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

## forward 迷惑性地不好用，建议改用 FWD 宏

众所周知，当你在转发一个“万能引用”参数时：

```cpp
template <class Arg>
void some_func(Arg &&arg) {
    other_func(arg);
}
```

如果此处 `arg` 传入的是右值引用，那么传入 `other_func` 就会变回左值引用了，不符合完美转发的要求。

因此引入了 `forward`，他会检测 `arg` 是否为“右值”：如果是，则 `forward` 等价于 `move`；如果不是，则 `forward` 什么都不做（默认就是左值引用）。

这弄得 `forward` 的外观非常具有迷惑性，又是尖括号又是圆括号的。

```cpp
template <class Arg>
void some_func(Arg &&arg) {
    other_func(std::forward<Arg>(arg));
}
```

实际上，forward 的用法非常单一：永远是 `forward<T>(t)` 的形式，其中 `T` 是 `t` 变量的类型。

又是劳保的魅力，利用同样是 C++11 的 `decltype` 就能获得 `t` 定义时的 `T`。

```cpp
void some_func(auto &&arg) {
    other_func(std::forward<decltype(arg)>(arg));
}
```

所以 `std::forward<decltype(arg)>(arg)` 实际才是 `forward` 的正确用法，只不过因为大多数时候你是模板参数 `Arg &&`，有的人偷懒，就把 `decltype(arg)` 替换成已经匹配好的模板参数 `Arg` 了，实际上是等价的。

这里需要复读 `arg` 太纱币了。实际上，我们可以定义一个宏：

```cpp
#define FWD(arg) std::forward<decltype(arg)>(arg)
```

这样就可以简化为：

```cpp
void some_func(auto &&arg) {
    other_func(FWD(arg));
}
```

少了烦人的尖括号，看起来容易懂多了。

> {{ icon.detail }} 但是，我们同学有一个问题，为什么 `std::forward` 要写成 `std::forward<T>` 的形式呢？为什么不是 `std::forward(t)` 呢？因为这样写的话，`forward` 也没法知道你的 `t` 是左是右了（函数参数始终会默认推导为左，即使定义的 `t` 是右）因此必须告诉 `forward`，`t` 的定义类型，也就是 `T`，或者通过 `decltype(t)` 来获得 `T`。

总之，如果你用的是 `auto &&` 参数，那么 `FWD` 会很方便（自动帮你 `decltype`）。但是如果你用的是模板参数 `T &&`，那么 `FWD` 也可以用，因为 `decltype(t)` 总是得到 `T`。

## bind 绑定成员函数是陋习，改用 lambda 或 bind_front

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

更好的是配合上文提到的 `FWD` 宏实现参数的完美转发：

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
            world(FWD(args)...); // 通过 FWD 完美转发给 world，避免引用退化
        };
        int x = 1;
        memfn(x, 2); // 双参数：OK
        memfn(3.14); // 单参数：OK
    }
}
```

同样可以定义一个称手的宏：

```cpp
#define BIND(func, ...) [__VA_ARGS__] (auto &&..._args) { func(FWD(_args)...); }
```

> {{ icon.tip }} 这里使用了宏参数包，此处 `__VA_ARGS__` 就是宏的 `...` 中的内容。注意区分宏的 `...` 和 C++ 变长模板的 `...` 是互相独立的。

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
        auto memfn = BIND(world, this);
        int x = 1;
        memfn(x, 2);
        memfn(3.14);
    }
}

int main() {
    // 捕获非 this 的成员函数也 OK：
    Class c;
    auto memfn = BIND(c.world, &c); // [&c] 按引用捕获 c 变量
    // 展开为：
    auto memfn = [&c] (auto &&..._args) { c.world(std::forward<decltype(_args)>(_args)...); }
    memfn(3.14);
}
```

> {{ icon.fun }} `BIND` 这个名字是随便取的，取这个名字是为了辱 `std::bind`。

为了解决 bind 不能捕获多参数重载的情况，C++17 还引入了 `std::bind_front` 和 `std::bind_back`，他们不需要 placeholder，但只能用于参数在最前或者最后的特殊情况。

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

## 救命！为什么我的全局函数不能作为函数对象？

当你的全局函数是模板函数，或带有重载的函数时：

```cpp
template <class T>
T square(T const t) {
    return t * t;
}

template <class Fn>
void do_something(Fn &&fn) {
    fn(2);
    fn(3.14);
}

int main() {
    do_something(square); // 编译错误：有歧义的重载
}
```

就会出现这样恼人的编译错误：

```
test.cpp: In instantiation of 'void do_something(Fn&&) [with Fn = T (*)(T) [with T = double]]':
test.cpp:18:21:   required from here
test.cpp:14:9: error: no matching function for call to 'do_something(<unresolved overloaded function type>)'
     do_something(square);
     ^~~~~~~~~~~~~
test.cpp:7:3: note: candidate: 'template<class Fn> void do_something(Fn&&) [with Fn = T (*)(T) [with T = double]]'
   void do_something(Fn &&fn) {
   ^~~~~~~~~~~~~
test.cpp:7:3: note:   template argument deduction/substitution failed:
test.cpp:14:21: note:   couldn't deduce template parameter 'Fn'
     do_something(square);
     ~~~~~~~~~~~~~^~~~~~
```

> {{ icon.detail }} 这是因为，模板函数和有重载的函数，是“多个函数对象”的“幻想联合体”，而 `do_something` 的 `Fn` 需要“单个”具体的函数对象。
>
> 一般来说是需要 `square<int>` 和 `square<double>` 才能变成“具体”的“单个”函数对象，传入 `do_something` 的 `Fn` 模板参数。
>
> 但是在“函数调用”的语境下，因为已知参数的类型，得益于 C++ 的“重载”机制，带有模板参数的函数，可以自动匹配那个模板参数为你参数的类型。
>
> 但现在你并没有指定调用参数，而只是指定了一个函数名 `square`，那 C++ “重载”机制无法确定你需要的是 `square<int>` 还是 `square<double>` 中的哪一个函数指针，他们的类型都不同，就无法具象花出一个函数对象类型 `Fn` 来，导致 `<unresolved overloaded function type>` 错误。

有趣的是，只需要套一层 lambda 就能解决：

```cpp
    do_something([] (auto x) { return square(x); }); // 编译通过
```

或者用我们上面推荐的 `BIND` 宏：

```cpp
#define FWD(arg) std::forward<decltype(arg)>(arg)
#define BIND(func, ...) [__VA_ARGS__] (auto &&..._args) { func(FWD(_args)...); }

    do_something(BIND(square)); // 编译通过
```

有时候，如果你想传递 `this` 的成员函数为函数对象，也会出现这种恼人的错误：

```cpp
struct Class {
    int func(int x) {
        return x + 1;
    }

    void test() {
        do_something(this->func); // 这里又会产生烦人的 unresolved overload 错误！
    }
};
```

同样可以包一层 lambda，或者用小彭老师提供的 `BIND` 宏，麻痹的编译器就不狗叫了：

```cpp
#define FWD(arg) std::forward<decltype(arg)>(arg)
#define BIND(func, ...) [__VA_ARGS__] (auto &&..._args) { func(FWD(_args)...); }

    void test() {
        do_something(BIND(func, this)); // 搞定
    }
```

> {{ icon.fun }} 建议修改标准库，把小彭老师这两个真正好用的宏塞到 `<utility>` 和 `<functional>` 里，作为 C++26 标准的一部分。

## 智能指针防止大对象移动

我们说一个类型大，有两种情况。

1. 类本身很大：例如 array
2. 类本身不大，但其指向的对象大，且该类是深拷贝，对该类的拷贝会引起其指向对象的拷贝：例如 vector

```cpp
sizeof(array<int, 1000>);  // 本身 4000 字节
sizeof(vector<int>);       // 本身 24 字节（成员是 3 个指针），指向的数组可以无限增大
```

> {{ icon.detail }} `sizeof(vector)` 为 24 字节仅为 `x86_64-pc-linux-gnu` 平台 `libstdc++` 库的实测结果，在 32 位系统以及 MSVC 的 Debug 模式 STL 下可能得出不同的结果，不可以依赖这个平台相关的结果来编程。

对于 vector，我们可以使用 `std::move` 移动语义，只拷贝该类本身的三个指针成员，而不对其指向的 4000 字节数组进行深拷贝。

对于 array，则 `std::move` 移动语义与普通的拷贝没有区别：array 作为静态数组容器，不是通过“指针成员”来保存数组的，而是直接把数组存在他的体内，对 array 的移动和拷贝是完全一样的！

总之，移动语义的加速效果，只对采用了“指针间接存储动态数据”的类型（如 vector、map、set、string）有效。对“直接存储静态大小数据”的类型（array、tuple、variant、成功“小字符串优化”的 string）无效。

所以，让很多“移动语义”孝子失望了：“本身很大”的类，移动和拷贝一样慢！

那么现在我们有个超大的类：

```cpp
using BigType = array<int, 1000>;  // 4000 字节大小的平坦类型

vector<BigType> arr;

void func(BigType x) {
    arr.push_back(std::move(x));  // 拷贝 4000 字节，超慢，move 也没用
}

int main() {
    BigType x;
    func(std::move(x));  // 拷贝 4000 字节，超慢，move 也没用
}
```

如何加速这种本身超大的变量转移？使用 `const` 引用：

```cpp
void func(BigType const &x)
```

似乎可以避免传参时的拷贝，但是依然不能避免 `push_back` 推入 `vector` 时所不得已的拷贝。

小技巧：改用 `unique_ptr<BigType>`

```cpp
using BigType = array<int, 1000>;  // 4000 字节大小的平坦类型

using BigTypePtr = unique_ptr<BigType>;

vector<BigType> arr;

void func(BigTypePtr x) {
    arr.push_back(std::move(x));  // 只拷贝 8 字节的指针，其指向的 4000 字节不用深拷贝了，直接移动所有权给 vector 里的 BigTypePtr 智能指针
    // 由于移走了所有权，x 此时已经为 nullptr
}

int main() {
    BigTypePtr x = make_unique<BigType>();  // 注意：用智能指针的话，需要用 make_unique 才能创建对象了
    func(std::move(x));  // 只拷贝 8 字节的指针
    // 由于移走了所有权，x 此时已经为 nullptr
}
```

上面整个程序中，一开始通过 `make_unique` 创建的超大对象，全程没有发生任何移动，避免了无谓的深拷贝。

对于不支持移动构造函数的类型来说，也可以用这个方法，就能在函数之间穿梭自如了。

```cpp
// 热知识：std::mutex 不支持移动

void func(std::mutex lock);

int main() {
    std::mutex lock;
    func(std::move(lock));  // 错误：mutex(mutex &&) = delete
}
```

```cpp
void func(std::unique_ptr<std::mutex> lock);

int main() {
    std::unique_ptr<std::mutex> lock = std::make_unique<std::mutex>();
    func(std::move(lock));  // OK：调用的是 unique_ptr(unique_ptr &&)，不关 mutex 什么事
}
```

更好的是 `shared_ptr`，连 `std::move` 都不用写，更省心。

```cpp
void func(std::shared_ptr<std::mutex> lock);

int main() {
    std::shared_ptr<std::mutex> lock = std::make_shared<std::mutex>();
    func(lock);  // OK：调用的是 shared_ptr(shared_ptr const &)，不关 mutex 什么事
    func(lock);  // OK：shared_ptr 的拷贝构造函数是浅拷贝，即使浅拷贝发生多次，指向的对象也不会被拷贝或移动
}
```

## optional 实现延迟初始化

假设我们有一个类，具有自定义的构造函数，且没有默认构造函数：

```cpp
struct SomeClass {
    int m_i;
    int m_j;

    SomeClass(int i, int j) : m_i(i), m_j(j) {}
};
```

当我们需要“延迟初始化”时怎么办？

```cpp
SomeClass c;
if (test()) {
    c = SomeClass(1, 2);
} else {
    c = SomeClass(2, 3);
}
do_something(c);
```

可以利用 optional 默认初始化为“空”的特性，实现延迟赋值：

```cpp
std::optional<SomeClass> c;
if (test()) {
    c = SomeClass(1, 2);
} else {
    c = SomeClass(2, 3);
}
do_something(c.value());  // 如果抵达此处前，c 没有初始化，就会报错，从而把编译期的未初始化转换为运行时异常
```

> {{ icon.story }} 就类似于 Python 中先给变量赋值为 None，然后在循环或 if 里条件性地赋值一样。

如果要进一步避免 `c =` 时，移动构造的开销，也可以用 `unique_ptr` 或 `shared_ptr`：

```cpp
std::shared_ptr<SomeClass> c;
if (test()) {
    c = std::make_shared<SomeClass>(1, 2);
} else {
    c = std::make_shared<SomeClass>(2, 3);
}
do_something(c);  // 如果抵达此处前，c 没有初始化，那么传入的就是一个 nullptr，do_something 内部需要负责检测指针是否为 nullptr
```

如果 `do_something` 参数需要的是原始指针，可以用 `.get()` 获取出来：

```cpp
do_something(c.get());  // .get() 可以把智能指针转换回原始指针，但请注意原始指针不持有引用，不会延伸指向对象的生命周期
```

> {{ icon.story }} 实际上，Java、Python 中的一切对象（除 int、str 等“钦定”的基础类型外）都是引用计数的智能指针 `shared_ptr`，只不过因为一切皆指针了，所以看起来好像没有指针了。

## if-auto 与 while-auto

需要先定义一个变量，然后判断某些条件的情况，非常常见：

```cpp
extern std::optional<int> some_func();

auto opt = some_func();
if (opt.has_value()) {
    std::cout << opt.value();
}
```

C++17 引入的 if-auto 语法，可以就地书写变量定义和判断条件：

```cpp
extern std::optional<int> some_func();

if (auto opt = some_func(); opt.has_value()) {
    std::cout << opt.value();
}
```

对于支持 `(bool)opt` 的 `optional` 类型来说，后面的条件也可以省略：

```cpp
extern std::optional<int> some_func();

if (auto opt = some_func()) {
    std::cout << opt.value();
}

// 等价于：
auto opt = some_func();
if (opt) {
    std::cout << opt.value();
}
```

类似的还有 while-auto：

```cpp
extern std::optional<int> some_func();

while (auto opt = some_func()) {
    std::cout << opt.value();
}

// 等价于：
while (true) {
    auto opt = some_func();
    if (!opt) break;
    std::cout << opt.value();
}
```

if-auto 最常见的配合莫过于 map.find：

```cpp
std::map<int, int> table;

int key = 42;
if (auto it = table.find(key); it != table.end()) {
    std::cout << it->second << '\n';
} else {
    std::cout << "not found\n";
}
```

## map + any 外挂属性

## 自定义 shared_ptr 的 deleter

## CHECK_CUDA 类错误检测宏

## 函数默认参数求值的位置是调用者

## 设置 locale 为 .utf8

## 花括号实现安全的类型转换检查

## 成员函数针对 this 的移动重载

## CHECK_CUDA 类错误检测宏

## 函数默认参数求值的位置是调用者

## 花括号实现安全的类型转换检查

## 临时右值转左值

C++ 有个特性：支持纯右值(prvalue)隐式转换成 const 的左值引用。

翻译：`int &&` 可以自动转换成 `int const &`。

```cpp
void func(int const &i);

func(1);  // OK：自动创建一个变量保存 1，然后作为 int const & 参数传入
```

实际上就等价于：

```cpp
const int tmp = 1;
func(tmp);
```

但是，`int &&` 却不能自动转换成 `int &`。

```cpp
void func(int &i);

func(1);  // 错误：无法从 int && 自动转换成 int &
```

> {{ icon.tip }} C++ 官方设置这个限制，是出于语义安全性考虑，因为参数接受 `int &` 的，一般都意味着这个是用作返回值，而如果 `func` 的参数是，`func(1)`。

为了绕开这个规则，我们可以定义一个帮手函数：

```cpp
T &temporary(T const &t) {
    return const_cast<T &>(t);
}

// 或者：
T &temporary(T &&t) {
    return const_cast<T &>(t);
}
```

然后，就可以快乐地转换纯右值为非 const 左值了：

```cpp
void func(int &i);

func(temporary(1));
```

> {{ icon.story }} 在 Libreoffice 源码中就有应用这个帮手函数。

> {{ icon.warn }} 临时变量的生命周期是一行

## ADL 机制

## shared_from_this

## requires 语法检测是否存在指定成员函数

## 设置 locale 为 .utf8 解决编码问题

## 成员函数针对 this 的移动重载

<!-- ## vector + unordered_map = LRU cache -->
<!--  -->
<!-- ## Lambda 捕获 unique_ptr 导致 function 报错怎么办 -->
<!--  -->
<!-- ## 多线程通信应基于队列，而不是共享全局变量 -->
<!--  -->
<!-- ## RAII 的 finally -->
<!--  -->
<!-- ## swap 缩小 mutex 区间代价 -->
