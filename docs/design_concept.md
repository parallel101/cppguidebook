# 鸭子类型与 C++20 concept (未完工)

[TOC]

如果一个东西叫起来像一只鸭，走起路来像一只鸭，那么不妨认为他就是一只鸭。

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

方案1：模板函数

```cpp
template <typename Dog>
void feeder(Dog dog) {
    dog.intro();
    dog.bark();
    dog.bark();
}
```

此处把 `Dog` 定义为模板函数

TODO
