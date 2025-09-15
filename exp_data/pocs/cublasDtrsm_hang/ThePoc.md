## Description

Hi，developers. There is an issue where a program cannot exit properly, regarding the "cublasDtrsm()" function !  I think this bug could potentially be used for a DoS attack.

## Replicate

I stumbled upon a bug while compiling a super simple `.cu` file. Here's what the `poc.cu` looks like:

```c
#include <stdio.h> 
#include <stdlib.h> 
#include <cuda_runtime.h> 
#include <cublas_v2.h> 

// Kernel function definition 
__global__ void mykernel(double *data, int numElements) { 
    int idx = threadIdx.x + blockIdx.x * blockDim.x; 
    if (idx < numElements) { 
        data[idx] *= 2; 
    } 
} 

int main() { 
    // Variable definitions and initialization 
    int numElements = 178; 
    int blockSize = 64; 
    size_t size = numElements * sizeof(double); 

    // Host memory allocation and initialization 
    double *h_data = (double *)malloc(size); 
    for (int i = 0; i < numElements; i++) { 
        h_data[i] = i; 
    } 

    // Device memory allocation 
    double *d_data = nullptr; 
    cudaMalloc((void**)&d_data, size); 

    // cuBLAS handle creation 
    cublasHandle_t handle; 
    cublasCreate(&handle); 

    // 1. Memory copy from host to device 
    cudaMemcpyAsync(d_data, h_data, size, cudaMemcpyHostToDevice, 0); 

    // 2. cuBLAS operation: cublasDtrsm 
    double alpha = 1.0; 
    cublasDtrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, \
CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, numElements, numElements, \
&alpha, d_data, numElements, d_data, numElements); 

    // 3. Kernel launch 
    int numBlocks = (numElements + blockSize - 1) / blockSize; 
    mykernel<<<numBlocks, blockSize>>>(d_data, numElements); 

    // 4. Memory copy from device to host 
    cudaMemcpyAsync(h_data, d_data, size, cudaMemcpyDeviceToHost, 0); 

    // 5. Device synchronization 
    cudaDeviceSynchronize(); 

    // Cleanup 
    cublasDestroy(handle); 
    free(h_data); 
    cudaFree(d_data); 

    return EXIT_SUCCESS; 
}
```

The compile command is: `nvcc poc.cu -o test -lcublas`  
Run it with: `./test`

## Enverments

it is GPU RTX4090 24G/

ubuntu20.04/

cpu Intel(R) Xeon(R) CPU E5-2680 v4 @ 2.40GHz/ 

nvidia driver 550.54, cuda version 12.4

(I only attempted it on a physical machine and did not test it in VM, such as docker)

## Impacts

* When I execute it, this program gets stuck for a long time and cannot exit normally. I had to hit `Ctrl+C` in the terminal to stop it. I use cuda-gdb to debug, but it still gets stuck. 

* After my multiple attempts, the main problem lies in the value of numElements. 178 happens to trigger it, but it cannot be slightly larger or smaller (such as 170 or 180).

* Even stranger is that if "numElements=170" is compiled to obtain the binary name 'test', and then "numElements=178" is compiled and overwritten with 'test', then this problem will not be triggered. If numElements=178 and a binary name is changed, such as test1, then this issue can still be triggered.  <u>If it cannot be triggered, just open a new terminal and change the binary name compiled. It should be able to trigger it. </u>

* I suspected that my function calls might not comply with the specifications, so I consulted the official manual. As follows:

```
cublasStatus_t cublasDtrsm(cublasHandle_t handle,
cublasSideMode_t side, cublasFillMode_t uplo,
cublasOperation_t trans, cublasDiagType_t diag,
int m, int n,
const double *alpha,
const double *A, int lda,
double *B, int ldb)
```

So I think there may be an issue with CublasDtrsm. 

* The compiled binary cannot be exited for a long time, as shown in the following figure:
  
  <img title="" src="https://github.com/MPSFuzz/images/blob/master/cublasDtrsm_hang_1.PNG?raw=true" alt="MPSFuzz/images" data-align="inline">

* cuda-gdb debug: It seems that there were some issues with the handling of d_data by cublasDrsm, which caused the subsequent mykernel function to remain stuck for a long time
  
  ![MPSFuzz/images](https://github.com/MPSFuzz/images/blob/master/cublasDstrsm_hang_2.PNG?raw=true)

## others

Thank you to your team for reviewing my submission. 

Have a nice day :) !
