#include <cuda_runtime.h>
#include "cudapp.cuh"
#include <span>

using namespace cudapp;

// half h; // 深度学习
// float f; // 图形学 50%
// double d; // 科研 50%

// GPU = 图形学 = float = 4 字节寄存器

__global__ void kernel(std::span<int> arr) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < arr.size(); i += gridDim.x * blockDim.x) {
        arr[i] = arr[i] * arr[i];
    }

    // SIMD

    // (0123435456..32) (0123435456..32) (0123435456..32) (0123435456..32)

    // thread 0: 0 100'0000 200'0000 ...
    // thread 1: 1 100'0001 200'0001 ...
    // thread 2: 2 100'0002 200'0002 ...
    // ...
    // thread 99'9999: 99'9999 199'9999 299'9999 ...

    // CPU AVX512: _mm_gather_ps -> 16 个 lane 分散访问
    // CPU AVX512: _mm_load_ps -> 16 个 lane 连续访问

    // wrap 0: thread 0~31: 0~31
    // wrap 1: thread 32~63: 1024~1055

    // thread 0~31: 0     // _mm_broadcast_ps
    // thread 0~31: 0~31  // _mm_load_ps
    // thread 0~31: 31~0  // _mm_shuffle_ps(_mm_load_ps)
    // thread 0~31: 0 2 4 6 8 // GPU ok, load
    // thread 0~31: 0 1231223 123123 54564569846 1273618923761 // _mm_gather_ps

    // blockDim = 1024
    // 1024 thread

    // 32 thread = 1 wrap
    // 1024 thread = 32 wrap

    // CPU 超线程 -> 1 核 同时处理 2 线程
    // thread 0 -> mem wait -> yield
    // thread 1 -> computing -> mem wait
    // CPU blockDim = 2

    // thread 0~31 = wrap 0 -> mem wait
    // thread 32~63 = wrap 1 -> computing -> mem wait
    // wrap 2
    // wrap 3

    // memory-bound -> 增加 thread 数量 -> 增大 blockDim
    // compute-bound（寄存器用量多）-> 减少 thread 数量 -> 减少 blockDim

    // SM (32 SIMD 处理器)
}

// CPU/GPU: 寄存器没有地址

// gridDim, blockDim
// grid的大小, block的大小
// grid里的block数量, 每个block里的thread数量
// blockIdx的范围, threadIdx的范围
// grid > block > thread

// CPU = 10 核心
// 核心 = 512 int 寄存器

// 显卡 = 1000 SM (Streaming Multiprocessor)
// SM = 50000 int 寄存器 (register)

// SM 最多能承受 1000 个 thread 可以，但是 100 个寄存器中，只有 49 个在物理寄存器里，还有 51 个，会被放到全局显存里，变成 memory-bound
// nvprof nvsight

// 如果 kernel 中，局部变量很多（每个 thread 用了很多寄存器）的话，就要调小 blockDim（每个 block 的 thread 数量）

// kernel 有 2 个 int 变量 (每个寄存器大小 4 字节)
// 每个 thread 会消耗 100 个寄存器 -> 50000 / 100 -> blockDim = 500

// 显卡 = grid
// grid 有 500 个 block
// 1000 SM

// gridDim = SM 数量 / 同时运行的内核数
// blockDim = 每个 SM 的寄存器数量 / kernel 中的局部变量数量（int 单位）

// blockDim 最好是 32 的整数倍
// 就算你是 1023 -> 32 wrap -> 1024 thread -> 最后一个 thread 作废
// blockDim 1024

int main() {
    CudaVector<int> arr;
    arr.resize(1000'0000, 2);

    CudaStream stream = CudaStream::Builder().build();

    // grid-stride-loop (网格跨步循环)

    CudaEvent t0 = stream.recordEvent();
    // floordiv
    // 100'0001 / 1000 = floor(1000.001) = 1000
    // ceildiv
    // (100'0001 + 999) / 1000 = ceil(1000.001) = 1001
    // (100'1000 + 999) / 1000 = 1001
    // kernel<<<(arr.size() + 999) / 1000, 1000, 0, stream>>>(arr);
    kernel<<<1000, 1000, 0, stream>>>(arr);
    CudaEvent t1 = stream.recordEvent();

    stream.join();

    printf("time diff brr: %f\n", t1 - t0);

    for (int i = 0; i < arr.size(); ++i) {
        if (arr[i] != 4) {
            printf("error at %d\n", i);
        }
    }

    return 0;
}
