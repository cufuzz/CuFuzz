#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

// CUDA API sequence to be tested
void UT(cusolverDnHandle_t *handle, cudaStream_t *stream, int *d_data) {
    // 1. Free device memory
    cudaFree(d_data);

    // 2. Destroy cuSolver handle
    cusolverDnDestroy(*handle);

    // 3. Destroy CUDA stream
    cudaStreamDestroy(*stream);

    // 4. Reset device
    cudaDeviceReset();
}

int main() {
    // Variable definitions and initialization
    int numElements = 256;
    size_t size = numElements * sizeof(int);

    // Device memory allocation
    int *d_data = nullptr;
    cudaMalloc((void**)&d_data, size);

    // cuSolver handle creation
    cusolverDnHandle_t handle;
    cusolverDnCreate(&handle);

    // Stream creation
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Execute the CUDA API sequence
    UT(&handle, &stream, d_data);

    return EXIT_SUCCESS;
}