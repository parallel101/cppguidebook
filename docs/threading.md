# C++ 多线程编程（未完工）
## 多线程中的常见概念
### 并行与并发

### 进程, 线程, 协程

### 线程的状态
## jthread
### 使用C++20的 `std::jthread`
 秉承学新不学旧的思路, 先介绍C++20提供的 `std::jthead`, 如果由于各种限制以至于读者无法使用jthread 也无需担心。 C++11提供的 `std::thread` 是 `std::jthead` 的阉割版。 你可以无缝的回到过去。
 ### 初始化`std::jthread`
C++20封装好的线程对象jthread接受一个**可调用对象(callable)**, 换句话说就是重载了 `()` 运算符的对象, 它可以是一个函数, 重载了 `()` 的类又或者是一个lambda表达式

同时jthread的构造函数本身就是有invoke功能, 所以第一个参数是可调用对象, 后面的参数直接跟上可调用对象的参数即可。 此时需要注意一点, 如果可调用对线的参数中有引用传递, 则需要用 `std::ref` 或者 `std::cref` 包装。 因为默认是按值或移动传递。

同时可调用对象还可以有返回, 但是jthread会忽略这个返回值。 如果想接住这个返回值需要借助 `std::future`。 见后。

jthread支持空初始化 `std::jthread jt;` 此时 `jt` 只是一个占位符, 并不是一个线程。如果后续需要分配任务, 使用jthread的移动语义。(jthread不能拷贝)
```cpp
std::jthread jt;
jt = std::jthread([] {});
// jt = std::move(other_jthread); 不能拷贝
```


## 线程的结束方式
线程的结束方式有两种：`join()` 和 `detach()`. 
其具体的关系如下: 
![[img/cpp_thread_join_detach.png]]
正如图中所示, 当 fork1_thread 调用其join()时, 其父线程必须等待其结束能继续进行后续操作. 而 fork2_thread 调用 detach() 时, 父线程则不需要等待他的结束. 当线程调用 detach() 必须要保证在 主线程main 结束之前结束, 不然main结束会释放资源, 如果此时子线程还没有结束则会导致使用一个已经被释放的资源。 

但是同时也要知道等待线程join可能是一件非常耗时的时候, 所以一般会在最后join。 但是detach()可以在一开始就进行, 因为反正也不需要等他返回。 

如果既不调用 `join()` 也不调用 `detach()`. 当线程对象的析构函数被调用时（通常在离开作用域或显式销毁时），由于线程对象仍然和一个活动的线程相关联，这会导致调用 `std::terminate()`，终止整个程序。

![[img/cpp_thread_join.png]]

实际上对于C++20的jthread而言, 会在其销毁的时候自动调用 `join()`. 但如果你使用的旧版本 `std::thread` 则需要手动的调用 `join()` 或者 `detach()`, 此时你应该通过thread_guard类保证在作用域结束之后自动调用的`join()`

> #### `joinable()`
> 初始化子线程 `t` 后, 该子线程自动就是 joinable 的, 也就是 `t.joinable()` 的值是 `true`. 换句话说 `t.joinable()` 等于 `true` 的条件就是该线程有一个与之相关联的线程(父线程)。 当其detach之后也就独立于父线程运行, 此时的 `t.joinable()` 就返回 `false`. 在官方的描述中,  `t.joinable()` 返回 `true` 则意味着可以通过 `get_id()` 得到这个线程的唯一标识. 但是当detach之后这个标识会返回0。换句话说, 一旦将一个线程detach之后就再也无法直接控制这个线程, 只能按照其原本的逻辑运行直至结束。 

在Unix语境下, detach的线程一般称作守护线程(daemon thread). 这类线程一般会在后台长时间运行. 虽然也可以将detach认为是即用即扔的线程, 但是要**保证其必须在main结束之前完成, 且不要持有资源**。

此时, 我们会发现：jthread是更加符合RAII思想的, 所以应该**优先使用jthread**。 同时如果你没有1000%的把握, 同时也为了维持你的san值处于正常水平, 赛博SCP基金会建议您**不要使用detach**。 

如果说用于初始化线程的可调用对象抛出异常但是没有处理时, 异常不会跨线程传播而是使用 `std::terminate()`。 如果内部处理了异常自然无事发生。 
```cpp
int main() {
	try { 
	/*
		根本捕获不到,而是直接 std::terminate()
		异常不会传播到主线程
	*/
		std::jthread thread([] {
			throw std::runtime_error("Error occurred");
		});
	}
	catch (const std::exception& e) {
		std::cout << "Caught exception: " << e.what() << std::endl;
	}
	std::jthread thread([] { 
		try { 
		// 在内部处理
			throw std::runtime_error("Error occurred"); 
		} catch (const std::exception& e) { 
			std::cout << "Caught exception in thread: " << e.what() << std::endl; 
		}
	});
	return 0;
} 
```
### jthread 的停止功能

### 无奈的妥协回到std::thread

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

## 小彭老师对话一则

关于 SharedPtr 的原子安全实现。

- 对话地址：https://github.com/parallel101/stl1weekend/issues/4
- 代码地址：https://github.com/parallel101/stl1weekend/blob/main/SharedPtr.hpp

sharedptr引用计数减被封装成如下的函数
```c++    
void _M_decref() noexcept {
        if (_M_refcnt.fetch_sub(1, std::memory_order_relaxed) == 1) {
            delete this;
        }
    }
```
该if块先判断原始引用计数是否等于1，如果为真则进行delete。然而判断和delete是两个操作，并不是一个原子操作。是否存在这样一种情况：判断条件成立，但在delete前有其它线程给引用计数+1？此时进行delete就出错了吧

**小彭老师**

没有问题的，因为fetch_sub返回1，实际上说明引用计数已经是0了，fetch_sub返回的是“旧值”，相当于后置i--，知道吧。如果已经为0，那就没有任何其他人持有该指针，我是独占的，那随便delete。

这样吧，我也听不懂你在讲什么，你来写一份你认为会产生问题的代码，让我分析。

**同学**

我设想了如下的代码：

```c++
shared_ptr<int> a = make_shared<int>();
void fun1() {
  a = nullptr;  // 析构a
}
void func2() {
  auto b = a; // 拷贝a到b
}

int main(){
  auto t1=std::thread(func1);
  auto t2=std::thread(func2);
  t1.join();
  t2.join();
  return 0;
}
```
在这段代码中，线程1析构a，而线程2拷贝a到b。由于多线程的缘故，我认为会出现以下的情况，线程1执行判断时

```c++
if (_M_refcnt.fetch_sub(1, std::memory_order_relaxed) == 1)
```
由于func2刚进入尚未执行拷贝，此时引用计数等于1还不是2，所以该判断为true。于是，线程1准备执行`delete this`将_SpCounter释放，就在这时线程2将func2彻底执行了，此时引用计数又从0变为了1，然而线程1并不知道这个变化，它仍然按照原本的轨迹去执行delete this。所以我认为这就出错了，而出错的原因是

```c++
void _M_decref() noexcept {
        if (_M_refcnt.fetch_sub(1, std::memory_order_relaxed) == 1) {
            delete this;
        }
    }
```
此处的判断和delete是两个操作，而非一个原子操作。

很抱歉之前未能及时回复

**小彭老师**

是的，这段代码有未定义行为！
然而 C++ 标准只要求了：
- 析构+拷贝 同时发生，是未定义行为。
- 拷贝+拷贝 同时发生，是安全的。
我的原子变量已经保证了 拷贝+拷贝 的安全，符合 C++ 标准的要求。
析构+拷贝 的情况，C++ 标准就并不要求安全，所以我的 shared_ptr 也没有责任去保证这种情况下的安全。

比如标准不要求 vector 的 clear 和 push_back 同时调用是线程安全的，那么我就不需要把 vector 实现为安全的。
如果标准规定了哪两个函数同时调用是安全的，我再去做。
比如标准就规定了 size 和 data 两个函数同时调用是线程安全的，我只需要符合这个就可以。
标准都没有规定必须安全的情况，我的容器如果产生未定义行为，我不负责任。

例如，C++ 标准对 `shared_ptr<T>` 的要求：
析构+拷贝 同时发生，是未定义行为。
拷贝+拷贝 同时发生，是安全的。
C++ 标准对 `atomic<shared_ptr<T>>` 的要求：
析构+拷贝 同时发生，是安全的。
拷贝+拷贝 同时发生，是安全的。
所以，只有当我是在实现atomic_shared_ptr时，才需要考虑你说的这种情况，而我现在实现的是shared_ptr，不需要考虑 析构+拷贝 的安全。

为什么拷贝+拷贝是安全的？我怎么没看到cppreference说？这很复杂，是另一句话里透露的通用规则，适用于所有容器，包括shared_ptr、unique_ptr、vector等全部的容器：
两个const成员函数，同时发生，没有未定义行为。
一个非const成员函数+一个const成员函数，同时发生，是未定义行为。
这句话自动适用于所有的容器了，所以你看到shared_ptr里没有说，但是我知道他是在另一个关于线程安全的页面上。

那么很明显，拷贝构造函数`shared_ptr(shared_ptr const &that)`是const的（对于被拷贝的that），而析构函数都是非const的，所以如果没有特别说明，一个容器同时调用拷贝+析构是未定义行为。而atomic_shared_ptr就属于特别说明了，所以他特别地同时访问const和非const函数是安全的。

完整的多线程安全规则表：
读+读=安全
读+写=未定义行为
写+写=未定义行为

所以实际上sharedptr所谓的“线程安全”，只不过是拷贝+拷贝这一情况的安全和拷贝+析构不同`shared_ptr`实例，同一个`shared_ptr`的并发非const访问是没保证的，`shared_ptr<T>`指向的那个`T`也是不保证的（由`T`的实现者“你”来保证）。
`shared_ptr`不是有三层吗？通俗的说就是他只需要保证中间这层控制块的线程安全性，不保证`shared_ptr`对象和`T`对象的安全性。
