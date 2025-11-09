# 结构体布局

## 认识基础类型

首先来认识一下，C++ 中常用的基础类型有：

| 类型 | 大小 | 值范围 |
| :-- | :-- | :-- |
| `bool` | 1 字节 | `true` 或 `false` |
| `int8_t` | 1 字节 | -128 到 127 |
| `int16_t` | 2 字节 | -32768 到 32767 |
| `int32_t` | 4 字节 | -2147483648 到 2147483647 |
| `int64_t` | 8 字节 | -2^63 到 2^63-1 |
| `uint8_t` | 1 字节 | 0 到 255 |
| `uint16_t` | 2 字节 | 0 到 65535 |
| `uint32_t` | 4 字节 | 0 到 4294967295 |
| `uint64_t` | 8 字节 | 0 到 2^64-1 |
| `float` | 4 字节 | 3.4e-38 到 3.4e+38 |
| `double` | 8 字节 | -1.79769e+308 到 1.79769e+308 |

## 认识结构体

### 结构体的好处

为什么要发明结构体呢？有什么好处？

例如，要表示一个点，需要 `x` 和 `y` 两个坐标。

假如，我们现在要实现很多用于处理点运算的函数，例如，计算两点间距离的函数。

如何把两个点的坐标传入函数呢？

用 `x0` 和 `y0` 表示第一个点的坐标，用 `x1` 和 `y1` 表示第二个点的坐标。

```cpp
double calc_distance(double x0, double y0, double x1, double y1) {
    double dx = x1 - x0;
    double dy = y1 - y0;
    return sqrt(dx * dx + dy * dy);
}
```

可以看到，我们想要传入两个点的坐标，但是函数却要传入四个参数，如果要传入更多的点坐标或其他复杂的抽象概念，就需要不断增加函数的参数，导致代码混乱，最终分不清哪个参数是哪个点的坐标。

因此，我们可以用结构体把两个坐标打包成一个整体：

```cpp
struct Point {
    double x;
    double y;
};
```

这样我们在编写和调用函数的时候，关注点就从每个点的具体坐标分量，转移到“点”这个抽象概念本身，不必纠结于这个点到底需要几个 `double` 来表示了。

```cpp
double calc_distance(Point p0, Point p1) {
    double dx = p1.x - p0.x;
    double dy = p1.y - p0.y;
    return sqrt(dx * dx + dy * dy);
}
```

这个函数现在一目了然，输入两个点，返回这两点间的距离，比之前四个参数看起来整洁多了。

#### 从属关系

得益于结构体的 `.` 级联访问，变量的从属关系也变得一目了然，`p0.x` 就是 0 号点的 x 坐标，`p1.x` 就是 1 号点的 x 坐标，再也不用一坨浆糊的 `x0`、`x1` 混搭风变量命名。

例如 `firstStudent.name` (第一个学生的名字) 和 `bestStudent.account.password` （最好学生的帐号的密码) 远比 `firstName` 和 `bestAccount1password` 直观。

> {{ icon.fun }} 这是的 `.` 就像中文 `的` 一样。

#### 结构体嵌套

谁说结构体只能由基本类型组成？谁说结构体里不能再含有结构体？

例如我们需要计算一条直线的长度，实现函数 `calc_line_length`，这时出现了和之前一样的抉择：

1. 把直线的两个端点 (用我们的 `Point` 类) 分别作为参数，传入函数：

```cpp
double calc_line_length(Point p0, Point p1) {
    return calc_distance(p0, p1);
}
```

2. 把直线这个抽象的概念封装成单独一个结构体，作为一个参数，传入函数：

```cpp
struct Line {
    Point p0;
    Point p1;
};

double calc_line_length(Line line) {
    return calc_distance(line.p0, line.p1);
}
```

`Line` 由两个 `Point` 组成，`Point` 又进一步由两个 `double` 组成。

想要理解 `Line` 的人只需要知道两个端点 `Point` 组成即可。
想要理解 `Point` 的人只需要知道两个 （或三个) 坐标值 `double` 组成即可。

这比一个 `Line` 直接用四个 `double` 来表示要直观得多。

如此层层封装，从基础类型建构出参天大树来，且每一层都以人类容易理解的方式，在保证可维护性的情况下，构造出复杂的多层逻辑。

> {{ icon.tip }} 研究表明，人类的大脑天生擅长理解**树状**组织的概念——他每一次学习，只需要理解每一层之间上下的关系，而不需要全部塞进大脑内存。而如果全部摊平，**扁平化**地堆在一起，脑容量就不够用，无法同时理解一切。

#### 方便升级

不仅如此，我们还很容易升级这个函数，例如当我们的甲方需求有变，现在需要处理三维点时，只需要往 Point 结构体里添加一个 `z` 成员：

```cpp
struct Point {
    double x;
    double y;
    double z;
};

double calc_distance(Point p0, Point p1) {
    double dx = p1.x - p0.x;
    double dy = p1.y - p0.y;
    double dz = p1.z - p0.z;
    return sqrt(dx * dx + dy * dy + dz * dz);
}
```

而函数的接口依然是两个 `Point` 作为参数不变，不用找到所有调用 `calc_distance` 的地方一个个去修改添加 `z0`、`z1` 了，只需要改函数实现本身即可。

### 结构体的布局

结构体的布局，是指结构体在内存中的布局方式。

例如，有如下结构体：

```cpp
struct Point {
    double x;
    double y;
};
```

这个结构体在内存中的布局，就是 `x` 和 `y` 两个成员挨在一起，没有空隙。

而如果有如下结构体：

```cpp
struct Point2 {
    double x;
    int y;
};
```
