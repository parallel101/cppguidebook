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
