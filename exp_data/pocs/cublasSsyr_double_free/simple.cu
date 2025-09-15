#include <fcntl.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <time.h>
#include <stdlib.h>

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

int main() {
    int loops = 0;
    srand((unsigned)time(NULL));
    while (loops < 2000) {
        
    //initialization
    printf("%d\n",loops);
    int rows = rand() % 257;
    int cols = rand() % 257;
    size_t size = rows * cols * sizeof(float);
    float *h_A = (float *)malloc(size);
    for (int i = 0; i < rows * cols; i++) {
        h_A[i] = (float)(i + 1);
    }
    //initialization

    printf("rows %d\n",rows);
    printf("cols %d\n",cols);
    printf("size %zu\n",size);


    // wrap all done!

    // Device memory pointer
    float *d_A = nullptr;

    // cuBLAS handle initialization
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Execute the CUDA API sequence

    cudaMalloc((void**)&d_A, size);

        // 执行 cuBLAS 操作
    float alpha = 1.0f;
    cublasSsyr(handle, CUBLAS_FILL_MODE_UPPER, rows, &alpha, d_A, 1, d_A, rows);
    cudaDeviceSynchronize();

    // UT(h_A, d_A, rows, cols, handle);

    // Cleanup
    free(h_A);
    cudaFree(d_A);
    cublasDestroy(handle);

   
    loops++;}
    return EXIT_SUCCESS;
}