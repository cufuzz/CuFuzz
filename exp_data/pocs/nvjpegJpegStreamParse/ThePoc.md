## Description

Hi，developers. There is a segmentation fault issue with the nvjpegJpegStreamParse function here!  I suspect there may be a potential memory issue.

## Replicate

I stumbled upon a bug while compiling a super simple `.cu` file. Here's what the `poc.cu` looks like:

```c
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <nvjpeg.h>

void UT(const unsigned char* jpegData, size_t jpegSize, \
nvjpegHandle_t nvjpegHandle, nvjpegJpegState_t nvjpegState, \
cudaStream_t stream) {
    if (jpegData == NULL || jpegSize == 0) return;

    nvjpegJpegStream_t jpegStream;
    nvjpegStatus_t status;
    unsigned int width, height;
    nvjpegChromaSubsampling_t subsampling;

    // Create a JPEG stream
    if (nvjpegJpegStreamCreate(nvjpegHandle, &jpegStream) != \
NVJPEG_STATUS_SUCCESS) {
        return;
    }

    // Parse the JPEG stream
    status = nvjpegJpegStreamParse(nvjpegHandle, jpegData, jpegSize, \
0, 0, jpegStream);
    if (status != NVJPEG_STATUS_SUCCESS) {
        nvjpegJpegStreamDestroy(jpegStream);
        return;
    }

    // Get frame dimensions
    status = nvjpegJpegStreamGetFrameDimensions(jpegStream, &width, \
&height);
    if (status != NVJPEG_STATUS_SUCCESS) {
        nvjpegJpegStreamDestroy(jpegStream);
        return;
    }

    // Get chroma subsampling
    status = nvjpegJpegStreamGetChromaSubsampling(jpegStream, \
&subsampling);
    if (status != NVJPEG_STATUS_SUCCESS) {
        nvjpegJpegStreamDestroy(jpegStream);
        return;
    }

    // Clean up
    nvjpegJpegStreamDestroy(jpegStream);
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        return EXIT_FAILURE;
    }

    const char* filename = argv[1];
    FILE* file = fopen(filename, "rb");
    if (!file) {
        return EXIT_FAILURE;
    }

    fseek(file, 0, SEEK_END);
    long fileSize = ftell(file);
    fseek(file, 0, SEEK_SET);

    unsigned char* jpgData = (unsigned char*)malloc(fileSize);
    if (!jpgData) {
        fclose(file);
        return EXIT_FAILURE;
    }

    size_t bytesRead = fread(jpgData, 1, fileSize, file);
    fclose(file);
    if (bytesRead != fileSize) {
        free(jpgData);
        return EXIT_FAILURE;
    }

    nvjpegHandle_t nvjpegHandle;
    nvjpegJpegState_t nvjpegState;
    cudaStream_t stream;

    // Initialize NVJPEG
    if (nvjpegCreate(NVJPEG_BACKEND_DEFAULT, NULL, &nvjpegHandle) \
!= NVJPEG_STATUS_SUCCESS) {
        free(jpgData);
        return EXIT_FAILURE;
    }
    if (nvjpegJpegStateCreate(nvjpegHandle, &nvjpegState) \
!= NVJPEG_STATUS_SUCCESS) {
        nvjpegDestroy(nvjpegHandle);
        free(jpgData);
        return EXIT_FAILURE;
    }
    if (cudaStreamCreate(&stream) != cudaSuccess) {
        nvjpegJpegStateDestroy(nvjpegState);
        nvjpegDestroy(nvjpegHandle);
        free(jpgData);
        return EXIT_FAILURE;
    }

    // Call UT function
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

* <img title="" src="https://github.com/MPSFuzz/images/blob/master/nvjpegJpegStreamParse_seg_fault.PNG?raw=true" alt="MPSFuzz/images" data-align="inline">

## others

Thank you to your team for reviewing my submission. 

Have a nice day :) !
