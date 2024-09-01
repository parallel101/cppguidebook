# C++ 多线程编程（未完工）

## 创建线程

TODO

## 为什么数据竞争

```cpp
if (table.count("小彭老师")) {
    return table.at("小彭老师");
} else {
    return 0;
}
```

```cpp
auto it = table.find("小彭老师");
if (it != table.end()) {
    return it->second;
} else {
    return 0;
}
```
