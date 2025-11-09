#include <cuda_runtime.h>
#include "cudapp.cuh"
#include <span>

using namespace cudapp;

int main() {
    CudaVector<int> arr;
    arr.resize(1000'0000, 2);

    CudaStream stream = CudaStream::Builder().build();

    CudaEvent t0 = stream.recordEvent();
    paralelFor<<<1000, 1000, 0, stream>>>(arr.size(), [arr = arr.data()] __device__ (int i) {
        arr[i] = arr[i] * arr[i];
    });
    CudaEvent t1 = stream.recordEvent();

    stream.join();

    printf("time diff: %f\n", t1 - t0);

    for (int i = 0; i < arr.size(); ++i) {
        if (arr[i] != 4) {
            printf("error at %d\n", i);
        }
    }

    return 0;
}
