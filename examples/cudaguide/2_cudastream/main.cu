#include <cuda_runtime.h>
#include "cudapp.cuh"
#include <span> // int *data, size_t size

using namespace cudapp;

// tuple array span pair string_view // 静态分配的类型，我称之为平板类型，可以进 GPU，可做 kernel 参数
// vector string map set             // 动态内存分配的，不能进 GPU，不能作为 kernel 参数

__global__ void kernel(std::span<int> arr) {
    int i = threadIdx.x; // 0, 1, 2, 3
    arr[i] = arr[i] * arr[i];
}

// CUDA C++ > PTX (IR) > SASS

// stream > event > block > thread

int main() {
    CudaVector<int> arr{1, 2, 3, 4};
    CudaVector<int> brr{2, 3, 4, 5, 6};
    CudaVector<int> crr{3, 4, 5, 6, 7, 8};

    CudaStream sa = CudaStream::Builder().build();
    CudaStream sb = CudaStream::Builder().build();

    CudaEvent before_arr = sa.recordEvent();
    kernel<<<1, 4, 0, sa>>>(arr);
    CudaEvent arr_done = sa.recordEvent();
    kernel<<<1, 6, 0, sa>>>(crr);
    CudaEvent crr_done = sa.recordEvent();

    CudaEvent before_brr = sb.recordEvent();
    kernel<<<1, 5, 0, sb>>>(brr);
    CudaEvent brr_done = sb.recordEvent();

    sa.join();
    sb.join();

    printf("time diff brr: %f\n", brr_done - before_arr);

    printf("arr:\n");
    for (int i = 0; i < arr.size(); ++i) {
        printf("%d\n", arr[i]);
    }
    printf("brr:\n");
    for (int i = 0; i < brr.size(); ++i) {
        printf("%d\n", brr[i]);
    }
    printf("crr:\n");
    for (int i = 0; i < crr.size(); ++i) {
        printf("%d\n", crr[i]);
    }

    return 0;
}
