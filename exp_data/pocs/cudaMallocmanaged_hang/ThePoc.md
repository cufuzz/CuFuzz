## Replicate

Hi, developers.

I stumbled upon a bug while compiling a super simple `.cu` file. Here's what the `poc.cu` looks like:

```c
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// define the kernel function
template <typename T>
__global__ void myKernel(T *data) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    data[idx] = idx;
}

int main() {
    int size = -2096128;    // ******   trigger the bug  ***************
    //int size = -11111;   //  It won't trigger
    //int size = 2147483647;  //   It won't trigger
    int blockSize = 4;

    int *managedData;
cudaMallocManaged(&managedData, size * sizeof(int));// ***trigger the bug

    int *hostData;
cudaMallocHost(&hostData, size * sizeof(int));

    int numBlocks = (size + blockSize - 1) / blockSize;
    myKernel<<<numBlocks, blockSize>>>(managedData);


    cudaFree(managedData);
cudaFreeHost(hostData);

    return EXIT_SUCCESS;
}
```

The compile command is: `nvcc poc.cu -o test`  
Run it with: `./test`

## Impacts

I know the `size` I gave is unreasonable, but normally `cudaMallocManaged` should validate the size. Even if it’s wrong, the program should still run without getting stuck for a long time. But that’s not what happened.

When I ran the program, it just wouldn’t terminate. I had to hit `Ctrl+C` in the terminal to stop it, and then I got a `watchdog: BUG: soft lockup` warning. It also interrupted other CUDA processes on my machine. My setup is an AMD 5995WX CPU, Ubuntu 20.04, and CUDA 12.7. Here’s the system log:

```shell
Mar 11 11:11:19 scu804 kernel: [5532472.170270] watchdog: BUG: soft lockup - CPU#39 stuck for 23s! [wrap:566263]
```

I also tried it on another machine with an Xeon CPU, Ubuntu 18.04, and CUDA 12.4. The program still wouldn’t terminate and had to be stopped manually with `Ctrl+C`, but I didn’t get the `soft lockup` warning this time.

## Cuda-gdb information

Here are some findings from my own attempts and debugging with `cuda-gdb`

* First, `cudaMallocManaged` does check the `size` itself. For example, if I set `size` to -11111 or 2147483647 (which are unreasonable values), the program still runs without crashing or throwing any errors. It just gives a warning when debugging with `cuda-gdb`. But only when `size` is set to -2096128, or values close to it, does the program get stuck for a long time.

* When I used `cuda-gdb` and checked the backtrace, it looks like the issue is coming from `cudaMallocManaged`, and it might be related to `libcuda.so` .
  
  ![](https://github.com/MPSFuzz/images/blob/master/image1.png?raw=true)

* I dug deeper into `libcuda.so` and noticed it uses `getpagesize()` from `libc.so`. When I disassembled the code, I found the issue at `test %rax, %rax`. I suspect it’s checking two values, and if they don’t match, it gets stuck in an infinite loop. Since CUDA is closed-source, I can only go this far.
  
  ![](https://github.com/MPSFuzz/images/blob/master/image2.png?raw=true)

## others

I think this bug could potentially be used for a DoS attack. If you’re unlucky, it might even cause bigger problems in certain environments, like what I saw on my Device 1. I hope to gain your attention

Have a nice day :) !
