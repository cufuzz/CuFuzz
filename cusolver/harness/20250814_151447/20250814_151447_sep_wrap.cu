#include <fcntl.h>
#include <cmath>
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include "/home/fanximing/cuda-graph-code/c_factors/mutate.h"


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
    int loops = 0;
    while (loops < 2000) {
    //initialization
    int numElements = 256;
    size_t size = numElements * sizeof(int);
    //initialization

    // wrap buffer
    u32 numElements_len = sizeof(numElements);
    u8 *numElements_buf = (u8 *)malloc(numElements_len);
    memcpy(numElements_buf, &numElements, numElements_len);

    // wrap havoc and writing .cur
    u32 numElements_len_havoc;
    numElements_len_havoc = random_havoc(numElements_buf, numElements_len, 0);
    numElements = *(int *)numElements_buf;
    numElements %= 4096;
    if (numElements<0) { numElements = -numElements ;}
    if (numElements==0) { numElements += 1 ;}
    FILE *file_1 = fopen("1.bin", "wb");
    fwrite(numElements_buf, numElements_len, 1, file_1);
    fclose(file_1);
    size =  numElements * sizeof(int);


    // wrap all done!

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

    free(numElements_buf);
    loops++;}
    return EXIT_SUCCESS;
}