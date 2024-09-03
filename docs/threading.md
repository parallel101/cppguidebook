# C++ 多线程编程（未完工）

## 创建线程
如果你上过《操作系统》这门课, 你一定听说过进程这个概念. 我们现在来区分这个 进程和线程这两个概念.
按照《操作系统》中的定义: 进程是操作系统资源分配的基本单位，而线程是处理器任务调度和执行的基本单位. 
翻译成人类的语言就是: **进程是一块地址空间, 线程是在这个空间上执行任务的实体**. 他们的关系类似于工地和在工地上的工人.

每一个进程中至少会有一个线程也就是父线程, 再通过这个父线程创建其他的子线程. 创建线程的操作使用了系统调用. 你不需要了解这些系统调用, 在C++中只标准库提供了 `<thread>`, 这是一个已经封装好统一接口.

> ##### (选读)关于Linux的线程状态(Windows真不熟)
> 
> 对于线程, 可以分成两类: 系统线程和用户线程, 系统线程是由操作系统创建的, 所以有系统的调度器管理. 但是用户线程系统是看不到的, 需要使用系统调用和系统线程才能让其被调度器接受. 但是C++标准库已经把这些系统调用都封装好了, C++使用者只需要调用标准库的提供的对象即可.
> 
> 一般来说(Linux), 线程会有以下状态:
> 
> ![[img/cpp_thread_thread_state_transform.png]]
> 
> 一个正常的流程是: 线程被创建(start), 然后就处于可执行状态(runnable), 现代CPU对于每个线程都是有时间片的, 运行完这次就运行下一个线程(也就是线程主动让出处理器`yield()`). 所以线程就会回到可以执行的(runable)状态.
> 
> 如果在执行过程中, 线程需要修改某个共享变量 (或者需要某个资源), 但是这个变量正在被别的线程修改(如果要修改共享变量需要提前上锁防止同时修改), 此时该线程就会等待(lock())进入Blocked状态或者Wait状态. 这个Blocked内部的两个小状态是等条件变量. 
> 对于synchronized join() 的情况一般是父线程在等待子线程的情况. 在Linux中, 这个Wait状态有两种情况: 可以被中断的(`TASK_INTERRUPTIBLE`)和不可被中断(`TASK_UNINTERRUPTIBLE`)的, 一个是收到系统的信号(事件还没发生)就不等了, 另一个是收到系统的信号(事件还没发生)也要等到事件发生.
> 
> 这里的stop状态和Wait状态类似都是等着. 但是stop的是因为系统给线程发送了某些信号(SIG有四个 `SIGSTOP`, `SIGTSTP`, `SIGTTIN`, `SIGTTOU`). 在线程收到系统的 `SIGCONT` 信号时就会结束等待. (这里的stop对应的是 `TASK_TRACED`, 还有一种是调试器(gdb,pdb之类的), 打个断点就是让其进入了 stop 状态)
> - **`SIGSTOP`**：强制暂停进程，无法捕获或忽略。
> - **`SIGTSTP`**：用户通过 `Ctrl+Z` 发送，用于临时暂停前台进程，可以被捕获处理。
> - **`SIGTTIN`**：后台进程试图读取终端输入时被暂停。
> - **`SIGTTOU`**：后台进程试图写入终端输出时被暂停。
> 
> 这里的状态在Linux中都存在进程描述符中. 或许你会疑惑为什么是进程描述符, 这是因为早期的Linux没有支持多线程, 对于Linux内核来说, 不区分进程和线程(都是 `task_struct`, 线程就是所谓的"轻量级进程")


 
 ## 初始化线程
C++封装好的线程对象thread接受一个**可调用对象(callable)**, 换句话说就是重载了 `()` 运算符的对象, 它可以是一个函数, 重载了 `()` 的类又或者是一个lambda表达式

同时thread的构造函数本身就是有invoke功能, 所以当你传入一个可调用对象时, 后面可以直接跟函数的参数. 



> [!note] 
> 在编译时，如果出现找不到 `pthread_create` 这个函数，原因是 `std::thread` 是基于pthread的所以需要在CMakeLists.txt中边界 `Threads::Threads`
>
>```cmake
>   find_package(Threads REQUIRED)
>   target_link_libraries(cpptest PUBLIC Threads::Threads)
>```
## 线程的结束方式
线程的结束方式有两种：`join()` 和 `detach()` 其具体的关系如下: 

![[img/cpp_thread_join_and_detach.png]]

正如图中所示, 当 fork1_thread 调用其join()时, 其父线程必须等待其结束能继续进行后续操作. 而 fork2_thread 调用 detach() 时, 父线程则不需要等待他的结束. 当线程调用 detach() 必须要保证在 主线程main 结束之前结束, 不然main结束会释放资源, 如果此时子线程还没有结束则会导致使用一个已经被释放的资源. 

但是同时也要知道等待线程join可能是一件非常耗时的时候, 所以一般会在最后join. 但是detach()可以在一开始就进行, 因为反正也不需要等他返回. 

如果既不调用 `join()` 也不调用 `detach()`. 当线程对象的析构函数被调用时（通常在离开作用域或显式销毁时），由于线程对象仍然和一个活动的线程相关联，这会导致调用 `std::terminate()`，终止整个程序。

![[img/cpp_thread_join.png]]

初始化子线程 `t` 后, 该子线程自动就是 joinable 的, 也就是 `t.joinable()` 的值是 `true`. 换句话说 `t.joinable()` 等于 `true` 的条件就是该线程有一个与之相关联的线程(父线程). 当其detach之后也就独立于父线程运行, 此时的 `t.joinable()` 就返回 `false`. 在官方的描述中,  `t.joinable()` 返回 `true` 则意味着可以通过 `get_id()` 得到这个线程的唯一标识. 但是当detach之后这个标识也就不存在了. 换句话说, 一旦将一个线程detach之后就再也无法直接控制这个线程, 只能按照其原本的逻辑运行直至结束. 

在Unix语境下, detach的线程一般称作守护线程(daemon thread). 这类线程一般会在后台长时间运行. 虽然也可以将detach认为是即用即扔的线程, 但是要保证其必须在main结束之前完成, 且不要持有资源.

## thread_guard
如果在启动线程之后, `join()` 之前, 如果发生异常则很有可能会漏掉 `join()`. 需要在 `catch` 中手动的 join. 这就又回到了C语言最初的 malloc-free 的问题上了. 所以封装一个类 `thread_guard`..

```cpp
class thread_guard{
	std::thread& t;\
public:
	explicit thread_guard(std::thread& t):t(t){}
	~thread_guard(){
		if(t.joinable()) t.join();
	}
	thread_guard(const thread_guard&) = delete;
	thread_guard& operator=(const thread_guard&) = delete;
};
```
使用的时候只要用 `thread_guard g(t);` 即可. 这样在退出main的代码块时, 自动调用 `thread_guard` 的析构函数. 

线程和 `unique_ptr` 一样是不能拷贝只能移动. 移动之后就原来线程死亡, 由新线程继续完成Task.(从上一个线程未完成的地方继续干). 

和 `thread_guard` 的思路一样, 移动之后的线程也要保证其可以自动的 `join()`..
```cpp
class scoped_thread{
	std::thread t;
public:
	explicit scoped_thread(std::thread t):t(std::move(t)){
		if(!t.joinable())
			throw std::logic_error("cant joinable()");
	}
	~scoped_thread(){
		t.join();
	}
	scoped_thread(const scoped_thread&) = delete;
	scoped_thread& operator=(const scoped_thread&) = delete;
};
```
所以一个完整的 thread_guard 如下：

```cpp
class thread_guard {
    std::thread& t_ref;  // 引用，管理左值时使用
public:
    // 接受左值引用的构造函数
    explicit thread_manager(std::thread& t_ref)
        : t_ref(t_ref) {
        if (!t_ref.joinable()) {
            throw std::logic_error("Thread is not joinable");
        }
    }
    // 接受右值的构造函数
    explicit thread_manager(std::thread&& t)
        : t_ref(t){
        if (!t_ref.joinable()) {
            throw std::logic_error("Thread is not joinable");
        }
    }
    // 析构函数：自动 join
    ~thread_manager() {
        if (t_ref.joinable()) {
            t_ref.join();
        }
    }
    // 禁用拷贝构造和拷贝赋值
    thread_manager(const thread_manager&) = delete;
    thread_manager& operator=(const thread_manager&) = delete;
    // 禁用移动构造和移动赋值
    thread_manager(thread_manager&&) = delete;
    thread_manager& operator=(thread_manager&&) = delete;
    std::thread&& get() { // 把线程吐出来
        return std::move(t_ref);
    }
    std::thread& get() const{
        return t_ref;
    }
};
```

## C++20 jthread
在C++20中, 自动join()已经整合进 `std::jthread` 中, 同时支持了停止这个操作, 相当于给thread加一个刹车.

具体实现通过两个类: `std::stop_source` 和 `std::stop_token`
- `stop_source` 是控制者(父线程)，它可以发出停止请求。
- `stop_token` 是被控制者(子线程)，它会感知到 `stop_source` 发出的停止请求。

![[img/cpp_thread_jthread_stop.png]]

简单来说 `stop_token`  会通过其成员函数 `stop_request()` 去检测 `stop_source` 有没有调用其成员函数 `request_stop()`. 所以必须把 `stop_token` 作为线程的可调用对象的一个参数传入.

`std::jthead` 内部管理了一个 `stop_source`, 你可以直接通过 `std::jthead` 对象直接调用其内部维护的 `stop_source.request_stop()`

节选cppref的案例:
```cpp
void finite_sleepy(std::stop_token stoken){
    for (int i = 10; i; --i){
        std::this_thread::sleep_for(300ms);
        if (stoken.stop_requested()){
            std::cout << "  困倦工人已被请求停止\n";
            return;
        }
        std::cout << "  困倦工人回去睡觉\n";
    }
}
int main(){
	std::jthread stop_worker(finite_sleepy);
	stop_worker.request_stop(); // 自己给自己批
	return 0;
}
```


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
