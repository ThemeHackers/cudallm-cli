#include <cuda_runtime.h>

extern "C" __global__ void simple_kernel() {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    (void)idx;
}

extern "C" __declspec(dllexport) void run_kernel() {
    simple_kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
}
