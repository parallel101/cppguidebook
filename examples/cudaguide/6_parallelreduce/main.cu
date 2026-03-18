#include <cuda_runtime.h>
#include "cudapp.cuh"

using namespace cudapp;

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
    paralelFor<<<1024, 1024, 0, stream>>>(arr.size(), [arr = arr.data()] __device__ (int i) {
    });
    CudaEvent t1 = stream.recordEvent();

    // 检查正确性
    paralelFor<<<1, 1, 0, stream>>>(1, [arr = arr.data(), n = int(arr.size())] __device__ (int i) {
    });

    stream.join();

    printf("time diff: %f ms\n", t1 - t0);

    return 0;
}
