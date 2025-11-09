#include <cuda_runtime.h>
#include <nvfunctional>
#include "cudapp.cuh" // 小彭老师现代 CUDA 框架，更符合现代 C++ 风格，减少官方 C 风格接口的繁琐

using namespace cudapp;

__global__ void kernel(int x) {
    printf("内核参数 x = %d\n", x);
    printf("线程编号 (%d, %d)\n", blockIdx.x, threadIdx.x);
}

int main() {
    // 启动内核的3种方式
    // 1. 官方三箭头语法糖（常用）
    // kernel<<<blockDim, gridDim, dynamicSmemBytes, stream>>>(...)
    int x = 42;
    kernel<<<3, 4, 0, 0>>>(x);

    // 2. cudaLaunchKernel
    void *args[] = {&x};
    CHECK_CUDA(cudaLaunchKernel(kernel, dim3(3), dim3(4), args, 0, 0));

    // 3. cudaLaunchKernelEx
    cudaLaunchConfig_t cfg{};
    cfg.blockDim = dim3(3); // threadIdx 的变化范围
    cfg.gridDim = dim3(4); // blockIdx 的变化范围
    cfg.dynamicSmemBytes = 0; // shared-memory 大小（暂不使用）
    cfg.stream = 0; // 在 0 号流（默认流）上启动
    cfg.attrs = nullptr;
    cfg.numAttrs = 0;
    CHECK_CUDA(cudaLaunchKernelEx(&cfg, kernel, x));

    // 还能查询内核名字
    const char *name;
    CHECK_CUDA(cudaFuncGetName(&name, kernel));
    printf("内核名字：%s\n", name);

    // 1. 强制同步：等待此前启动过的所有内核执行完成
    CHECK_CUDA(cudaDeviceSynchronize());

    // 2. 仅同步 0 号流（默认流）：等待此前在 0 号流上启动过的所有内核执行完成
    CHECK_CUDA(cudaStreamSynchronize(0));

    // 3. 仅同步 0 号流，但使用小彭老师现代 CUDA 框架
    CudaStream::defaultStream().join();

    return 0;
}
