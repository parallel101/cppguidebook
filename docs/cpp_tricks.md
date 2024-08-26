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

TODO

## RAII 地分配一段内存空间

所谓“内存空间”实际上就是“char 数组”。

TODO

## 函数参数也可以 auto

TODO

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

即使你想要默认值 0 这一特性，这也比 `[]` 更好，因为 `[]` 的默认值是会对 table 做破坏性修改的。

```cpp
return table["小彭老师"]; // 如果"小彭老师"这一键不存在，会创建"小彭老师"并设为默认值 0
```

```cpp
const map<string, int> table;
return table["小彭老师"]; // 编译器报错：[] 需要非 const 的 map 对象，因为他会破坏性修改
```

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

## const 居然应该后置...

```cpp
const int *p;
int *const p;
```

你能看出来上面这个 const 分别修饰的是谁吗？

1. 指针指向的 `int`
2. 指针本身 `int *`

```cpp
const int *p;  // 1
int *const p;  // 2
```

为了看起来更加明确，我通常都会后置所有的 const 修饰。

```cpp
int const *p;
int *const p;
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
const int *p = &i;
*p = 1;  // 错误：p 指向的对象不可变
p = &j;  // OK：p 本身可变，可以改变指向
```

> {{ icon.tip }} `int const *` 和 `const int *` 等价！只有 `int *const` 是不同的。

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

TODO

## RAII 的 finally

## swap 缩小 mutex 区间代价

## map + any 外挂属性

## bind 是历史糟粕，应该由 Lambda 表达式取代

## forward 迷惑性地不好用，建议改用 FWD 宏

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

```cpp
struct SomeClass {
    int m_i;
    int m_j;

    SomeClass(int i, int j) : m_i(i), m_j(j) {}
};
```

## if-auto 与 while-auto

TODO

## 临时右值转左值

C++ 有个特性：支持纯右值(prvalue)隐式转换成 const 的左值引用。

翻译：`int &&` 可以自动转换成 `int const &`。

```cpp
void func(int const &i);

func(1);  // OK：自动创建一个变量保存 1，然后作为 int const & 参数传入

// 等价于：
const int tmp = 1;
func(tmp);
```

但是，`int &&` 却不能自动转换成 `int &`。

```cpp
void func(int &i);

func(1);  // 错误：无法从 int && 自动转换成 int &
```

> {{ icon.tip }} 设置这个限制，可能是出于语义安全性考虑，因为参数接受 `int &` 的，一般都意味着这个是用作返回值，而如果 `func` 的参数是，`func(1)`。

## vector + unordered_map = LRU cache

## 多线程通信应基于队列，而不是共享全局变量

## 自定义 shared_ptr 的 deleter

## Lambda 捕获 unique_ptr 导致 function 报错怎么办

## CHECK_CUDA 类错误检测宏

## 位域（bit-field）

## 设置 locale 为 .utf8

TODO
