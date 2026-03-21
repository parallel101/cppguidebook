#include <cuda_runtime.h>
#include "cudapp.cuh"

using namespace cudapp;

// grid > block > thread

// SM -> 1 wrap = 32 thread
// 1 队伍 = 32 士兵
// 走的时候，齐步走 (类似于 CPU SIMD)

// 31 士兵需要走，1 个被 mask
// 32 都被 mask -> 队伍直接废掉，不走了


// __shared__ 1024 int
// __shared__ = max 65536 int = L1
// SM -> block 1 -> block 2

// 1024 thread -> 32 wrap

// wrap 0 -> __syncthreads
// wrap 1 -> __syncthreads
// wrap 2 -> __syncthreads

// ........

// ........


// .......1

// .......1 暂停 ........ 暂停 .......3 都到齐了吧！再一起走！


// __shared__:
// ........                    ........ -> read

// SM -> wrap 0 -> 如需等待，则暂停此 wrap，执行剩下不需要等待


// __shared__ = L1
// arr: 0000000 xxxxxxxxxx
// arr: 0123456 xxxxxxxxxx
// arr: 6543210 xxxxxxxxxx
// arr: 0246802 xxxxxxxxxx
// arr: 4813245 xxxxxxxxxx
// bank conflict xxxxxx

// GlobalMemory -> prefetch 预取 -> L1 -> reg calc
// arr: 0000000 xxxxx
// arr: 0123456 xxxx
// arr: 6543210 xxx
// arr: 0246802 xx
// arr: 4813245 x

// coleasing 裂开的访问x

// 0-32 64-71
// SM -> block -> warp 1 2 8 1

// xxxx + xxx
// xxxx + xxxx


__global__ void paralelReverse1(int *arr, size_t n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n / 2) {
        return;
    }

    int tmp = arr[i];
    arr[i] = arr[n - 1 - i];
    arr[n - 1 - i] = tmp;
}

__global__ void paralelReverse2(int *arr, size_t n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n / 2) {
        return;
    }

    int j = ((gridDim.x * 2) - 1 - blockIdx.x) * blockDim.x + threadIdx.x;

    // wrap 0 -> buf[0-31] = 0
    // wrap 1 -> buf[32-63] = 1
    // wrap 2 -> buf[64-...] = 2

    __shared__ int buf_i[1024];
    __shared__ int buf_j[1024];

    buf_i[threadIdx.x] = arr[i];
    buf_j[threadIdx.x] = arr[j];
    __syncthreads();

    arr[j] = buf_i[1023 - threadIdx.x];
    arr[i] = buf_j[1023 - threadIdx.x];

    // gridDim = 2
    // blockDim = 1024

    // 0-1023   1024-2047
    // 2047-1024   1023-0
}

// 星际穿越之类的
// warp

//  ??     ???     32       1
// grid > block > warp > thread

// warp-intrinstics

__global__ void paralelReverse3(int *arr, size_t n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n / 2) {
        return;
    }

    // 32-bit unsigned int
    // 0xffffffff


    int wrapIdx = (threadIdx.x % 32);
    int tmp = arr[n - 32 + wrapIdx];
    arr[n - 32 + wrapIdx] = __shfl_sync(0xffffffff, arr[i], 31 - wrapIdx);
    arr[i] = __shfl_sync(0xffffffff, tmp, 31 - wrapIdx);


    // _mm512_shuffle_ps(_mm512_load_ps(arr[i:i+32]), _MM_SHUFLE(31, ..., 3, 2, 1, 0));

    // // thread 0:
    // int val = __shfl_sync(__activemask(), arr[0], 31);
    // // thread 1:
    // __shfl_sync(__activemask(), arr[1], 30);
    // // thread 2:
    // __shfl_sync(__activemask(), arr[2], 29);
    // // thread 31:
    // __shfl_sync(__activemask(), arr[31], 0);

    // __warp_shared__ int shflbuf[32];
    // shflbuf[threadIdx] = arr[i];
    // arr[i] = shflbuf[offset];
}

int main() {
    CudaVector<int> arr;
    arr.resize(65536 * 1024); // 0 1 2 3 4 ...

    CudaStream stream = CudaStream::Builder().build();

    // 初始化
    paralelFor<<<1024, 1024, 0, stream>>>(arr.size(), [arr = arr.data()] __device__ (int i) {
        arr[i] = i;
    });

    // 计算内核
    CudaEvent t0 = stream.recordEvent();
    paralelReverse3<<<(arr.size() / 1024) / 2, 1024, 0, stream>>>(arr.data(), arr.size());
    CudaEvent t1 = stream.recordEvent();

    // 检查正确性
    // paralelFor<<<1024, 1024, 0, stream>>>(arr.size(), [arr = arr.data(), n = int(arr.size())] __device__ (int i) {
    //     if (arr[i] != n - 1 - i) {
    //         printf("error at %d: %d != %d\n", i, arr[i], n - 1 - i);
    //     }
    // });
    // 100000000 999999999 .... 4 3 2 1 0

    stream.join();

    printf("time diff: %f ms\n", t1 - t0);

    return 0;
}
