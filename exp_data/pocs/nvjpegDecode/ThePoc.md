## Description

Hi，developers. There is a segmentation fault issue with the nvjpegDecoder function here!  I suspect there may be a potential memory issue.

## Replicate

I stumbled upon a bug while compiling a super simple `.cu` file. Here's what the `poc.cu` looks like:

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <nvjpeg.h>

void UT(const unsigned char* jpegData, size_t jpegSize, nvjpegHandle_t nvjpegHandle, nvjpegJpegState_t nvjpegState, cudaStream_t stream) {
    if (jpegData == NULL || jpegSize == 0) return;

    int nComponents = 0;
    nvjpegChromaSubsampling_t subsampling;
    int widths[NVJPEG_MAX_COMPONENT] = {0};
    int heights[NVJPEG_MAX_COMPONENT] = {0};

    nvjpegStatus_t status = nvjpegGetImageInfo(nvjpegHandle, jpegData, jpegSize, &nComponents, &subsampling, widths, heights);
    if (status != NVJPEG_STATUS_SUCCESS) {
        return;
    }

    int out_channels = 3;
    int out_width = widths[0];
    int out_height = heights[0];

    nvjpegImage_t out_img;
    memset(&out_img, 0, sizeof(out_img));
    
    // Allocate device memory for each channel
    for (int i = 0; i < out_channels; i++) {
        if (cudaMalloc(&out_img.channel[i], out_width * out_height) != cudaSuccess) {
            // Free any already allocated channels
            for (int j = 0; j < i; j++) {
                cudaFree(out_img.channel[j]);
            }
            return;
        }
        out_img.pitch[i] = out_width;
    }

    status = nvjpegDecode(nvjpegHandle, nvjpegState, jpegData, jpegSize, NVJPEG_OUTPUT_RGB, &out_img, stream);
    if (status != NVJPEG_STATUS_SUCCESS) {
        for (int i = 0; i < out_channels; i++) {
            cudaFree(out_img.channel[i]);
        }
        return;
    }
    
    cudaStreamSynchronize(stream);

    // Allocate host memory and copy one channel
    unsigned char* host_channel = (unsigned char*)malloc(out_width * out_height);
    if (host_channel) {
        cudaMemcpy(host_channel, out_img.channel[0], out_width * out_height, cudaMemcpyDeviceToHost);
        
        // Do something with host_channel here...
        
        free(host_channel);
    }

    // Free device memory
    for (int i = 0; i < out_channels; i++) {
        cudaFree(out_img.channel[i]);
    }
}

int main(int argc, char* argv[]) {
    const char* filename;
    if (argc > 1) {
        filename = argv[1];
    } else {
        return EXIT_FAILURE;
    }

    nvjpegHandle_t nvjpegHandle;
    nvjpegJpegState_t nvjpegState;
    cudaStream_t stream;

    // Initialize resources
    if (nvjpegCreate(NVJPEG_BACKEND_DEFAULT, NULL, &nvjpegHandle) != NVJPEG_STATUS_SUCCESS) {
        return EXIT_FAILURE;
    }
    if (nvjpegJpegStateCreate(nvjpegHandle, &nvjpegState) != NVJPEG_STATUS_SUCCESS) {
        nvjpegDestroy(nvjpegHandle);
        return EXIT_FAILURE;
    }
    if (cudaStreamCreate(&stream) != cudaSuccess) {
        nvjpegJpegStateDestroy(nvjpegState);
        nvjpegDestroy(nvjpegHandle);
        return EXIT_FAILURE;
    }

    // Open and read file
    FILE* file = fopen(filename, "rb");
    if (!file) {
        nvjpegJpegStateDestroy(nvjpegState);
        nvjpegDestroy(nvjpegHandle);
        cudaStreamDestroy(stream);
        return EXIT_FAILURE;
    }

    // Get file size
    fseek(file, 0, SEEK_END);
    long fileSize = ftell(file);
    fseek(file, 0, SEEK_SET);

    // Read file data
    unsigned char* jpgData = (unsigned char*)malloc(fileSize);

    size_t bytesRead = fread(jpgData, 1, fileSize, file);
    fclose(file);

    if (bytesRead != fileSize) {
        free(jpgData);
        nvjpegJpegStateDestroy(nvjpegState);
        nvjpegDestroy(nvjpegHandle);
        cudaStreamDestroy(stream);
        return EXIT_FAILURE;
    }

    // Process the image
    UT(jpgData, fileSize, nvjpegHandle, nvjpegState, stream);

    // Cleanup
    free(jpgData);
    cudaStreamDestroy(stream);
    nvjpegJpegStateDestroy(nvjpegState);
    nvjpegDestroy(nvjpegHandle);

    return EXIT_SUCCESS;
}
```

The compile command is: `nvcc poc.cu -o test -lnvjpeg`  
Run it with: `./test 1.bin`

"1.bin" is an input that can trigger an issue, which is mutated from a normal JPG image. Unfortunately, this 1. bin cannot be opened as a jpg properly because the mutation has disrupted the structure

## Enverments

it is GPU RTX4090 24G/

ubuntu20.04/

cpu Intel(R) Xeon(R) CPU E5-2680 v4 @ 2.40GHz/ 

nvidia driver 550.54, cuda version 12.4

(I only attempted it on a physical machine and did not test it in VM, such as docker)

## debug information

* I executed this binary directly in the terminal using a compute optimizer and CUDA GDB, and the result is shown in the following figure:

* <img title="" src="https://github.com/MPSFuzz/images/blob/master/nvjpegDecode_seg_fault.PNG?raw=true" alt="MPSFuzz/images" data-align="inline">
  
  

## others

Thank you to your team for reviewing my submission. 

Have a nice day :) !
