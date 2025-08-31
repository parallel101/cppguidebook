#include <cstdio>
#include <thread>

// 类似于以下 CPU 函数的效果
void kernel_cpu() {
    #pragma omp parallel for collapse(2)
    for (int blockIdx = 0; blockIdx < 3; blockIdx++) {
        for (int threadIdx = 0; threadIdx < 4; threadIdx++) {
            printf("func 调用\n");
            printf("线程编号 (%d, %d)\n", blockIdx, threadIdx);
        }
    }
}

int main() {
    printf("正在后台启动内核\n");

    // 启动后台线程，让线程在后台默默执行，不会阻塞主线程
    std::thread th(kernel_cpu);

    printf("主线程继续照常执行\n");

    // 强制同步：等待此前启动过的计算线程执行完成
    th.join();

    printf("后台线程已经把内核执行完毕\n");

    return 0;
}
