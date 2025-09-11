import re

class PROMPTS():
    def __init__(self):
        self.prompt_call_order = "You are an expert agent specialized in cuda programming. \
                Your task is to extract the order in which CUDA APIs appear within each code segment, from a given guide document by user. \
                If a CUDA API calls another CUDA API, class, or structure, identify such calls.  \
                Save the aforementioned calling relationships in a JSON format, and save APIs order in another JSON.  \
                After receiving the user's prompt text, you first need to extract the code from it, express the code in the standard format of C++ code, \
                which may consist of multiple segments. Then, extract the following key information from each segment of code. \
                The description in the user prompt about code could help you extract following key.\
                The text provided by the user comes from a PDF-format book, where a code example may span across pages, \
                marked with '(continues on next page)' and '(continued from previous page)' as connecting signs. \
                If you receive a pair of these signs, then consider the two pieces of code as a whole for analysis. \
                If you only receive a single '(continues on next page)' sign, then ignore the piece of code before the sign and do not analyze it. \
                You must generate the output to save the calling relationships in a JSON containing a list with JOSN objects having the following keys: \
                'head', 'head_type', 'description', 'relation', 'tail', and 'tail_type'. 'Head' is just a \
                CUDA api name referenced within the examples extracted from the user-provided document. 'Head_type' is the type of 'head', \
                which can be one of __global__, __host__, or __device__. \
                'description' is a simple sentence for describing the 'head' function. \
                'Tail' can be also extracted from the user-provided document and is a CUDA api or a CUDA class(like c++ class). \
                'Relation' represents the relationship between 'head' and 'tail', it is 'calls', 'calls' means that 'head' calls 'tail'. \
                'tail' can be a function argument of 'head', or it can be explicitly called within the body of the 'head'.\
                'Tail' must conform to the 'relation' with 'head'. 'Tail_type' is the specific type of 'tail', it could be CUDA_API, CUDA_CLASS. \
                If the function represented by 'head' does not call any other CUDA functions, \
                then the 'relation' is marked as 'single', and both 'tail' and 'tail_type' are empty.\
                There is neither need to analyze the main function nor the functions defined by customer. Just analyze the entries by CUDA \
                The response should follow the format of using Jison between ```json and ```. \
                Then you must generate the output to save the APIs order in a JSON containing several lists where every list repersents an api order of one code segment, as follows: \
                {'order':[[cuda_api_1, cuda_api_2, ...], [cuda_api_3, cuda_api_1,...] ,...]}. Consider only sequences that contain two or more CUDA APIs; \
                if a code segment contains only one CUDA API, do not include it in the JSON.\
                This response about API order also is saved as Jison format between ```json and ```.\
                Don't have any other statements between ```json and ``` to break the json format"

        self.prompt_calls = "You are an expert agent specialized in cuda programming. \
                Your task is to extract the relationship of CUDA API calls within each code segment, from a given guide document by user. \
                If a CUDA API calls another CUDA API, class, or structure, identify such calls.  \
                Save the aforementioned calling relationships in a JSON format.  \
                After receiving the user's prompt text, you first need to extract the code from it, \
                which may consist of multiple segments. Then, extract the following key information from each segment of code. \
                The description in the user prompt about code could help you extract following key.\
                You must generate the output to save the calling relationships in a JSON containing a list with JOSN objects having the following keys: \
                'head', 'head_type', 'description', 'relation', 'tail', and 'tail_type'. 'Head' is just a \
                CUDA api name referenced within the examples extracted from the user-provided document. 'Head_type' is the type of 'head', \
                which can be one of __global__, __host__, or __device__. \
                'description' is a simple sentence for describing the 'head' function. \
                'Tail' can be also extracted from the user-provided document and is a CUDA api or a CUDA class(like c++ class). \
                'Relation' represents the relationship between 'head' and 'tail', it is 'calls', 'calls' means that 'head' calls 'tail'. \
                'tail' can be a function argument of 'head', or it can be explicitly called within the body of the 'head'.\
                'Tail' must conform to the 'relation' with 'head'. 'Tail_type' is the specific type of 'tail', it could be CUDA_API, CUDA_CLASS. \
                If the function represented by 'head' does not call any other CUDA functions, \
                then the 'relation' is marked as 'single', and both 'tail' and 'tail_type' are empty.\
                The description in the user prompt about code could help you extract key information.\
                The text provided by the user comes from a PDF-format book, where a code example may span across pages, \
                marked with '(continues on next page)' and '(continued from previous page)' as connecting signs. \
                If you receive a pair of these signs, then consider the two pieces of code as a whole for analysis. \
                If you only receive a single '(continues on next page)' sign, then ignore the piece of code before the sign and do not analyze it. \
                There is neither need to analyze the main function nor the functions defined by customer. Just analyze the entries by CUDA \
                The response should follow the format of using Jison between ```json and ```. \
                Don't have any other statements between ```json and ``` to break the json format"

        self.prompt_order = "You are an expert agent specialized in cuda programming. \
                Your task is to extract the order in which CUDA APIs appear within each code segment, from a given guide document by user. \
                Save the aforementioned APIs order in JSON format.  \
                After receiving the user's prompt text, you first need to extract the code from it, \
                which may consist of multiple segments. Then, extract the following key information from each segment of code. \
                You must generate the output to save the APIs order in a JSON containing several lists where every list repersents an api order of one code segment, as follows: \
                'order':[[cuda_api_1, cuda_api_2, ...], [cuda_api_3, cuda_api_1,...] ,...]}. Consider only sequences that contain two or more CUDA APIs; \
                if a code segment contains only one CUDA API, do not include it in the JSON.\
                The description in the user prompt about code could help you extract key information.\
                The text provided by the user comes from a PDF-format book, where a code example may span across pages, \
                marked with '(continues on next page)' and '(continued from previous page)' as connecting signs. \
                If you receive a pair of these signs, then consider the two pieces of code as a whole for analysis. \
                If you only receive a single '(continues on next page)' sign, then ignore the piece of code before the sign and do not analyze it. \
                There is neither need to analyze the main function nor the functions defined by customer. Just analyze the entries by CUDA \
                This response about API order also is saved as Json format between ```json and ```.\
                Don't have any other statements between ```json and ``` to break the json format"

        self.prompt_call_from_cuda_sample_only_cu = "You are an expert agent specialized in cuda programming. \
                Your task is to extract the relationship of CUDA API calls within the code segment, from a given context file by user. \
                If a CUDA API calls another CUDA API, class, or structure, identify such calls.  \
                Save the aforementioned calling relationships in a JSON format.  \
                You must generate the output to save the calling relationships in a JSON containing a list with JOSN objects having the following keys: \
                'head', 'head_type', 'description', 'relation', 'tail', and 'tail_type'. 'Head' is just a \
                CUDA api name referenced within the examples extracted from the user-provided code. 'Head_type' is the type of 'head', \
                which can be one of __global__, __host__, or __device__. \
                'description' is a simple sentence for describing the 'head' function. \
                'Tail' can be also extracted from the user-provided code and is a CUDA api or a CUDA class(like c++ class, c struct). \
                'Relation' represents the relationship between 'head' and 'tail', it is 'calls', 'calls' means that 'head' calls 'tail'. \
                'tail' can be a function argument of 'head', or it can be explicitly called within the body of the 'head'.\
                'Tail' must conform to the 'relation' with 'head'. 'Tail_type' is the specific type of 'tail', it could be CUDA_API, CUDA_CLASS. \
                If the function represented by 'head' does not call any other CUDA functions, \
                then the 'relation' is marked as 'single', and both 'tail' and 'tail_type' are empty.\
                There is neither need to analyze the main function nor the functions defined by customer. Just analyze the entries by CUDA，do not concern about the micro 'CUDA_CHECK'. \
                The response should follow the format of using Jison between ```json and ```. \
                Don't have any other statements between ```json and ``` to break the json format"

        self.prompt_order_from_cuda_sample_only_cu = "You are an expert agent specialized in cuda programming. \
                Your task is to extract the order in which CUDA APIs appear within the code segment, from a given context file by user. \
                Save the aforementioned APIs order in JSON format.  \
                You must generate the output to save the APIs order in a JSON containing several lists where every list repersents an api order of the code context, as follows: \
                'order':[[cuda_api_1, cuda_api_2, ...], [cuda_api_3, cuda_api_1,...] ,...]}. Consider only sequences that contain two or more CUDA APIs; \
                if the code context contains only one CUDA API, do not include it in the JSON.\
                There is neither need to analyze the main function nor the functions defined by customer. Just analyze the entries by CUDA, do not concern about the micro 'CUDA_CHECK'. \
                This response about API order is saved as Json format between ```json and ```.\
                Don't have any other statements between ```json and ``` to break the json format"

        self.gen_harness_with_a_api_senquence = "You are an expert agent specialized in cuda programming. \
                Your task is to generate a cuda program using the given cuda api sequence by user.  \
                You must call the APIs in the order they are given, and you can also have the previous API call the next one.  \
                You should try to ensure that the code can be compiled smoothly."

nvjpeg_gain_api_sig_from_txt = """
    Read the given nvjpeg tutorial file to obtain the APIs and corresponding API signatures. 
    Extract only signed APIs.
    If there is no function signature in the given file, return empty content
    
    You must generate the output in a JSON containing a list with JOSN objects having the following keys:
    api_name and api_signature. 
    api_name is the API of the extracted signed nvjgeg library. Pay attention to removing line breaks, extra spaces, and semicolons.
    api_signature is the corresponding function signature of api_name, do not include any explanatory content, only include signatures
    This response is saved as Json format between ```json and ```.
    Don't have any other statements between ```json and ``` to break the json format
    """

cublas_cunpp_gain_api_sig_from_pdf_seg = """
    Read the given tutorial file to obtain the APIs and corresponding API signatures. 
    Extract only signed APIs.
    If there is no function signature in the given file, return empty content

    You must generate the output in a JSON containing a list with JOSN objects having the following keys:
    api_name and api_signature. 
    api_name is the API of the extracted signed nvjgeg library. Pay attention to removing line breaks, extra spaces, and semicolons.
    api_signature is the corresponding function signature of api_name, do not include any explanatory content, only include signatures
    This response is saved as Json format between ```json and ```.
    Don't have any other statements between ```json and ``` to break the json format
    """

send_request_code_prompt_4rt = """
        You are a CUDA program expert. Please write a CUDA program that includes the int main() function and void UT() function. 
        Among them, main() contains definitions and initialization of some variables, and UT() contains the CUDA API sequence to be tested,
        and receives some variables defined in main() as arguments. 
        
        The given CUDA API sequence is %s, try to call in the given API order as much as possible in the UT().
        There are known API call relationships here: %s.  You should try you best to have the previous API call the next one, according to given call relationships.
        The UT() should define the necessary custom kernel function(e.g., __device__, __global__) which will be used by a CUDA API.
        If all given CUDA APIs do not call the kernel, the kernel function could be omitted.
        
        The signatures of some functions are as follows: %s. 
                       
        You should pay special attention to the initialization of some variables, especially CUDA variables or structures,
        because incorrect initialization can cause segmentation fault when API calls to these variables in the future.
        Try to initialize the handle using the ` cudaXxxCreate () ` function or other certain method instead of directly assigning it to NULL/0
        
        When passing some variables to UT(), determine whether to use '&' based on the complete usage of the variable, such as UT(&team). 
        If using UT(Steam), the stream is only a copy within UT(), and releasing the stream in main will result in an error. Because the stream was free before it was assigned a value.
        
        If it involves defining an array, malloc must be used to dynamically allocate memory and initialize it. 
        Cannot use fixed length static initialization, such as float A[2] = {1,2}
        
        Format requirements : The code should follow the C or C++ code specification, and the program code should be complete and properly formatted.
        Please note to include the necessary header files.
        In the code, you should write a long sentence without using line breaks, avoiding the newline character \ n.
        Do not print content in the generated code, such as using 'printf', 'fprintf', or 'std:cout' et.al.
        Don’t make up APIs that don't exist.

        Here is a template function that you can refer to its format:

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Kernel function definition
__global__ void mykernel(int *data, int numElements) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < numElements) {
        data[idx] *= 2;
    }
}

// CUDA API sequence to be tested
void UT(int *h_data, int *d_data, int numElements, int blockSize) {
    // 1. Memory copy from host to device
    size_t size = numElements * sizeof(int);
    cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);
    
    // 2. Kernel launch
    int numBlocks = (numElements + blockSize - 1) / blockSize;
    mykernel<<<numBlocks, blockSize>>>(d_data, numElements);
    
    // 3. Memory copy from device to host
    cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);
    
    // 4. Device synchronization
    cudaDeviceSynchronize();
}

int main() {
    // Variable definitions and initialization
    int numElements = 256;
    int blockSize = 16;
    size_t size = numElements * sizeof(int);
    
    // Host memory allocation and initialization
    int *h_data = (int *)malloc(size);
    for (int i = 0; i < numElements; i++) {
        h_data[i] = i;
    }
    
    // Device memory allocation
    int *d_data = nullptr;
    cudaMalloc((void**)&d_data, size);
    
    // Execute the CUDA API sequence
    UT(h_data, d_data, numElements, blockSize);
    
    // Cleanup
    free(h_data);
    cudaFree(d_data);
    
    return EXIT_SUCCESS;
}
		"""

send_request_code_prompt_4nvjpeg = """
		You are a CUDA program expert. Please write a CUDA program that includes the int main() function and void UT() function. 
        Among them, main() contains definitions and initialization of some variables, and UT() contains the CUDA API sequence to be tested,
        and receives some variables defined in main() as arguments. 
        In the main(), add image reading part, i.e., use unsigned char* jpgData = (unsigned char*)malloc(fileSize); and
        size_t bytesRead = fread(jpgData, 1, fileSize, file); to read one image.
        Subsequent UT() may not require an image, so no operation can be performed after reading
        Read only one image, if multiple images are required in UT(), reuse the one read.
        
        The given CUDA API sequence is %s, try to call in the given API order as much as possible in the UT().
        There are known API call relationships here: %s. You should try you best to have the previous API call the next one, according to given call relationships.
        The UT() should define the necessary custom kernel function(e.g., __device__, __global__) which will be used by a CUDA API.
        If all given CUDA APIs do not call the kernel, the kernel function could be omitted.
        
        The signatures of some functions are as follows: %s.
                       
        You should pay special attention to the initialization of some variables, especially CUDA variables or structures,
        because incorrect initialization can cause segmentation fault when API calls to these variables in the future.
        
        Try to initialize the handle using the ` nvjpegXxxCreate () ` function instead of directly assigning it to NULL/0
        
        When passing some variables to UT(), determine whether to use '&' based on the complete usage of the variable, such as UT(&team). 
        If using UT(Steam), the stream is only a copy within UT(), and releasing the stream in main will result in an error. Because the stream was free before it was assigned a value.

        Be careful not to repeatedly free or destroy certain variables in main() and UT(), 
        such as nvjpegJpegStateDestroy(nvjpegState); if they appear in both main() and UT(), it will lead to a segmentation fault.

        Cannot set fixed values for the test file (such as the width and height of the image), as the input file may vary.
    
        Format requirements : The code should follow the C code specification, and the program code should be complete and properly formatted.
        Please note to include the necessary header files.
        In the code, you should write a long sentence without using line breaks, avoiding the newline character \ n.
        Do not print content in the generated code, such as using 'printf', 'fprintf', or 'std:cout' et.al.
        Don’t make up APIs that don't exist.

        Here is a template function that you can refer to its format:

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <nvjpeg.h>

void UT(const unsigned char* jpegData, size_t jpegSize, 
        nvjpegHandle_t nvjpegHandle, nvjpegJpegState_t nvjpegState, 
        cudaStream_t stream) {
    if (jpegData == NULL || jpegSize == 0) return;

    int nComponents = 0;
    nvjpegChromaSubsampling_t subsampling;
    int widths[NVJPEG_MAX_COMPONENT] = {0};
    int heights[NVJPEG_MAX_COMPONENT] = {0};

    nvjpegStatus_t status = nvjpegGetImageInfo(nvjpegHandle, jpegData, jpegSize,
                                             &nComponents, &subsampling, widths, heights);
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

    status = nvjpegDecode(nvjpegHandle, nvjpegState, jpegData, jpegSize,
                        NVJPEG_OUTPUT_RGB, &out_img, stream);
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
        cudaMemcpy(host_channel, out_img.channel[0],
                 out_width * out_height, cudaMemcpyDeviceToHost);
        
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
		"""

send_request_code_prompt_4cunpp = """
		You are a CUDA program expert. Please write a CUDA program that includes the int main() function and void UT() function. 
        Among them, main() contains definitions and initialization of some variables, and UT() contains the CUDA API sequence to be tested,
        and receives some variables defined in main() as arguments. 
        In the main(), add image reading part, i.e., use unsigned char* jpgData = (unsigned char*)malloc(fileSize); and
        size_t bytesRead = fread(jpgData, 1, fileSize, file); to read one image.
        Subsequent UT() may not require an image, so no operation can be performed after reading
        Read only one image, if multiple images are required in UT(), reuse the one read.
        
        The given CUDA API sequence is %s, try to call in the given API order as much as possible in the UT().
        There are known API call relationships here: %s. You should try you best to have the previous API call the next one, according to given call relationships.
        The UT() should define the necessary custom kernel function(e.g., __device__, __global__) which will be used by a CUDA API.
        If all given CUDA APIs do not call the kernel, the kernel function could be omitted.
        
        The signatures of some functions are as follows: %s.
                       
        You should pay special attention to the initialization of some variables, especially CUDA variables or structures,
        because incorrect initialization can cause segmentation fault when API calls to these variables in the future.
        
        Try to initialize the handle using the ` nvjpegXxxCreate () ` function instead of directly assigning it to NULL/0
        
        When passing some variables to UT(), determine whether to use '&' based on the complete usage of the variable, such as UT(&team). 
        If using UT(Steam), the stream is only a copy within UT(), and releasing the stream in main will result in an error. Because the stream was free before it was assigned a value.

        Be careful not to repeatedly free or destroy certain variables in main() and UT(), 
        such as nvjpegJpegStateDestroy(nvjpegState); if they appear in both main() and UT(), it will lead to a segmentation fault.

        Cannot set fixed values for the test file (such as the width and height of the image), as the input file may vary.
    
        Format requirements : The code should follow the C code specification, and the program code should be complete and properly formatted.
        Please note to include the necessary header files.
        In the code, you should write a long sentence without using line breaks, avoiding the newline character \ n.
        Do not print content in the generated code, such as using 'printf', 'fprintf', or 'std:cout' et.al.
        Don’t make up APIs that don't exist.

        Here is a template function that you can refer to its format:

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <nvjpeg.h>

void UT(const unsigned char* jpegData, size_t jpegSize, 
        nvjpegHandle_t nvjpegHandle, nvjpegJpegState_t nvjpegState, 
        cudaStream_t stream) {
    if (jpegData == NULL || jpegSize == 0) return;

    int nComponents = 0;
    nvjpegChromaSubsampling_t subsampling;
    int widths[NVJPEG_MAX_COMPONENT] = {0};
    int heights[NVJPEG_MAX_COMPONENT] = {0};

    nvjpegStatus_t status = nvjpegGetImageInfo(nvjpegHandle, jpegData, jpegSize,
                                             &nComponents, &subsampling, widths, heights);
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

    status = nvjpegDecode(nvjpegHandle, nvjpegState, jpegData, jpegSize,
                        NVJPEG_OUTPUT_RGB, &out_img, stream);
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
        cudaMemcpy(host_channel, out_img.channel[0],
                 out_width * out_height, cudaMemcpyDeviceToHost);
        
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
		"""


send_request_code_prompt_4cublas = """
        You are a CUDA program expert. Please write a CUDA program that includes the int main() function and void UT() function. 
        Among them, main() contains definitions and initialization of some variables, and UT() contains the CUDA API sequence to be tested,
        and receives some variables defined in main() as arguments. 

        The given CUDA API sequence is %s, try to call in the given API order as much as possible in the UT().
        There are known API call relationships here: %s.  You should try you best to have the previous API call the next one, according to given call relationships.
        The UT() should define the necessary custom kernel function(e.g., __device__, __global__) which will be used by a CUDA API.
        If all given CUDA APIs do not call the kernel, the kernel function could be omitted.

        The signatures of some functions are as follows: %s. 

        You should pay special attention to the initialization of some variables, especially CUDA variables or structures,
        because incorrect initialization can cause segmentation fault when API calls to these variables in the future.
        Try to initialize the handle using the ` cudaXxxCreate () ` function or other certain method instead of directly assigning it to NULL/0

        When passing some variables to UT(), determine whether to use '&' based on the complete usage of the variable, such as UT(&team). 
        If using UT(Steam), the stream is only a copy within UT(), and releasing the stream in main will result in an error. Because the stream was free before it was assigned a value.
        
        If it involves defining an array, malloc must be used to dynamically allocate memory and initialize it. 
        Cannot use fixed length static initialization, such as neither float A[2] = {1,2} nor float A[2][2] = {{1,1},{2,2}} 
        
        Format requirements : The code should follow the C or C++ code specification, and the program code should be complete and properly formatted.
        Please note to include the necessary header files.
        In the code, you should write a long sentence without using line breaks, avoiding the newline character \ n.
        Do not print content in the generated code, such as using 'printf', 'fprintf', or 'std:cout' et.al.
        Don’t make up APIs that don't exist.

        Here is a template function that you can refer to its format:

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Kernel function definition
__global__ void mykernel(int *data, int numElements) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < numElements) {
        data[idx] *= 2;
    }
}

// CUDA API sequence to be tested
void UT(int *h_data, int *d_data, int numElements, int blockSize) {
    // 1. Memory copy from host to device
    size_t size = numElements * sizeof(int);
    cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);

    // 2. Kernel launch
    int numBlocks = (numElements + blockSize - 1) / blockSize;
    mykernel<<<numBlocks, blockSize>>>(d_data, numElements);

    // 3. Memory copy from device to host
    cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);

    // 4. Device synchronization
    cudaDeviceSynchronize();
}

int main() {
    // Variable definitions and initialization
    int numElements = 256;
    int blockSize = 16;
    size_t size = numElements * sizeof(int);

    // Host memory allocation and initialization
    int *h_data = (int *)malloc(size);
    for (int i = 0; i < numElements; i++) {
        h_data[i] = i;
    }

    // Device memory allocation
    int *d_data = nullptr;
    cudaMalloc((void**)&d_data, size);

    // Execute the CUDA API sequence
    UT(h_data, d_data, numElements, blockSize);

    // Cleanup
    free(h_data);
    cudaFree(d_data);

    return EXIT_SUCCESS;
}

    When generating cuBLAS code, note a critical difference in matrix storage:  
    C/C++ uses row-major order, while cuBLAS defaults to column-major order (inherited from Fortran).  
    To handle this:  
    1. If your input matrix is row-major, use `CUBLAS_OP_T` (transpose) to treat it as column-major.  
    2. Set `lda` (leading dimension) to the number of rows in column-major storage.  
    
    A case is as follow:
    
#include <cublas_v2.h>
#include <stdio.h>

int main() {
    // Example: Multiply row-major matrices A (3x2) * B (2x3) = C (3x3)
    // ---- Step 1: Define row-major matrices in C ----
    float *A = (float *)malloc(3 * 2 * sizeof(float));
    A[0 * 2 + 0] = 1; A[0 * 2 + 1] = 4;  // A[0][0], A[0][1]
    A[1 * 2 + 0] = 2; A[1 * 2 + 1] = 5;  // A[1][0], A[1][1]
    A[2 * 2 + 0] = 3; A[2 * 2 + 1] = 6;  // A[2][0], A[2][1]
    
    float *B = (float *)malloc(2 * 3 * sizeof(float));
    B[0 * 3 + 0] = 1; B[0 * 3 + 1] = 2; B[0 * 3 + 2] = 3;  // B[0][0], B[0][1], B[0][2]
    B[1 * 3 + 0] = 4; B[1 * 3 + 1] = 5; B[1 * 3 + 2] = 6;  // B[1][0], B[1][1], B[1][2]

                     
    float *C = (float *)calloc(3 * 3, sizeof(float));  // calloc 自动初始化为 0

    // ---- Step 2: cuBLAS setup ----
    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1.0, beta = 0.0;

    // Copy matrices to device (as row-major)
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, 3 * 2*sizeof(float));
    cudaMalloc(&d_B, 2 * 3*sizeof(float));
    cudaMalloc(&d_C, 3 * 3*sizeof(float));
    cublasSetMatrix(3, 2, sizeof(float), A, 3, d_A, 3); // lda=3 (rows)
    cublasSetMatrix(2, 3, sizeof(float), B, 2, d_B, 2); // lda=2 (rows)

    // ---- Step 3: Compute C = A * B ----
    // Since A and B are row-major, use transposed flags: 
    // A^T (3x2 -> column-major) * B^T (3x2 -> column-major) = C^T
    cublasSgemm(handle, 
                CUBLAS_OP_T,    // Transpose A (row->col)
                CUBLAS_OP_T,    // Transpose B (row->col)
                3, 3, 2,        // M, N, K (C is 3x3)
                &alpha, 
                d_A, 3,         // lda = rows of A (original)
                d_B, 2,         // ldb = rows of B (original)
                &beta, 
                d_C, 3);        // ldc = rows of C (3)

    // ---- Step 4: Verify ----
    cublasGetMatrix(3, 3, sizeof(float), d_C, 3, C, 3);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cublasDestroy(handle);
    return 0;
}
    
		"""

send_request_code_prompt_4cufft = """
        You are a CUDA program expert. Please write a CUDA program that includes the int main() function and void UT() function. 
        Among them, main() contains definitions and initialization of some variables, and UT() contains the CUDA API sequence to be tested,
        and receives some variables defined in main() as arguments. 

        The given CUDA API sequence is %s, try to call in the given API order as much as possible in the UT().
        There are known API call relationships here: %s.  You should try you best to have the previous API call the next one, according to given call relationships.
        The UT() should define the necessary custom kernel function(e.g., __device__, __global__) which will be used by a CUDA API.
        If all given CUDA APIs do not call the kernel, the kernel function could be omitted.

        The signatures of some functions are as follows: %s. 

        You should pay special attention to the initialization of some variables, especially CUDA variables or structures,
        because incorrect initialization can cause segmentation fault when API calls to these variables in the future.
        Try to initialize the handle using the ` cudaXxxCreate () ` function or other certain method instead of directly assigning it to NULL/0

        When passing some variables to UT(), determine whether to use '&' based on the complete usage of the variable, such as UT(&team). 
        If using UT(Steam), the stream is only a copy within UT(), and releasing the stream in main will result in an error. Because the stream was free before it was assigned a value.
        
        If it involves defining an array, malloc must be used to dynamically allocate memory and initialize it. 
        Cannot use fixed length static initialization, such as neither float A[2] = {1,2} nor float A[2][2] = {{1,1},{2,2}} 
        
        Format requirements : The code should follow the C or C++ code specification, and the program code should be complete and properly formatted.
        Please note to include the necessary header files.
        In the code, you should write a long sentence without using line breaks, avoiding the newline character \ n.
        Do not print content in the generated code, such as using 'printf', 'fprintf', or 'std:cout' et.al. 
        Don’t make up APIs that don't exist.

        Here is a template function that you can refer to its format:

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Kernel function definition
__global__ void mykernel(int *data, int numElements) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < numElements) {
        data[idx] *= 2;
    }
}

// CUDA API sequence to be tested
void UT(int *h_data, int *d_data, int numElements, int blockSize) {
    // 1. Memory copy from host to device
    size_t size = numElements * sizeof(int);
    cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);

    // 2. Kernel launch
    int numBlocks = (numElements + blockSize - 1) / blockSize;
    mykernel<<<numBlocks, blockSize>>>(d_data, numElements);

    // 3. Memory copy from device to host
    cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);

    // 4. Device synchronization
    cudaDeviceSynchronize();
}

int main() {
    // Variable definitions and initialization
    int numElements = 256;
    int blockSize = 16;
    size_t size = numElements * sizeof(int);

    // Host memory allocation and initialization
    int *h_data = (int *)malloc(size);
    for (int i = 0; i < numElements; i++) {
        h_data[i] = i;
    }

    // Device memory allocation
    int *d_data = nullptr;
    cudaMalloc((void**)&d_data, size);

    // Execute the CUDA API sequence
    UT(h_data, d_data, numElements, blockSize);

    // Cleanup
    free(h_data);
    cudaFree(d_data);

    return EXIT_SUCCESS;
}

		"""



harness_separate_code_prompt_4rt = """
            You are a CUDA program expert. Reorganize the code without changing the semantics of the code. 
            I want to test the code with different inputs, like fuzz testing. 
            Therefore, firstly, you should find some initialization data that can be changed in the code only in the main(), and do not find some fixed value initialization.
            Do not find variables that are calculated based on other variables.
            If the length of a variable depends on other variables, but the specific values of its elements do not depend on other variables, 
            it is also should be included.

            Next, reorganize the main() function into two parts. The first part initializes the variables you found in the previous step, each variable on a separate line. 
            The first part is surrounded by a pair of comments '//initialization'.

            Please note that some simple memory allocation operations (i.e. only allocating memory to arrays without initializing with specific values)
            do not be included in the '//initialization' process.

            The second part is the rest of the main() function.

            Format requirements : The code should follow the C or C++ code specification, and the program code should be completed by NVCC and properly formatted.
            For long code statements, there is no need to wrap them, write them in one line.

            Here is a example that you can refer to its format:

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void mykernel(int *data, int numElements) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < numElements) {
        data[idx] *= 2;
    }
}

void UT(int *h_data, int *d_data, int numElements, int blockSize) {
    size_t size = numElements * sizeof(int);
    cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);
    int numBlocks = (numElements + blockSize - 1) / blockSize;
    mykernel<<<numBlocks, blockSize>>>(d_data, numElements);
    cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
}

int main() {
    //initialization
    int numElements = 256;
    int blockSize = 16;
    int *h_data = (int *)malloc(numElements * sizeof(int));
    for (int i = 0; i < numElements; i++) {
        h_data[i] = i;
    //initialization
    
    }

    int *d_data = nullptr;
    cudaMalloc((void**)&d_data, numElements * sizeof(int));
    UT(h_data, d_data, numElements, blockSize);
    free(h_data);
    cudaFree(d_data);
    return EXIT_SUCCESS;
}

            The code to be refactored is as follows:
            %s
    		"""


harness_separate_code_prompt_4nvjpeg = """
            You are a CUDA program expert. Reorganize the code without changing the semantics of the code. 
            I want to test the code with different inputs, like fuzz testing. 
            Therefore, firstly, you should find some initialization data that can be changed in the code only in the main(), and do not find some fixed value initialization.
            Do not find variables that are calculated based on other variables.
            If the length of a variable depends on other variables, but the specific values of its elements do not depend on other variables, 
            it is also should be included.
            For read image files, the read char* jpgData should be treated as a mutable variable rather than a file name.
            
            Next, Separate an independent part from main(). The part initializes the variables you found in the previous step, each variable on a separate line. 
            The part is surrounded by a pair of comments '//initialization'.

            Please note that some simple memory allocation operations (i.e. only allocating memory to arrays without initializing with specific values)
            do not be included in the '//initialization' process.
            
            The remaining parts of main() remain unchanged.

            Format requirements : The code should follow the C code specification, and the program code should be completed by NVCC and properly formatted.
            For long code statements, there is no need to wrap them, write them in one line.

            Here is a example that you can refer to its format:

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <nvjpeg.h>

void UT(const unsigned char* data, size_t jpegSize, 
        nvjpegHandle_t nvjpegHandle, nvjpegEncoderState_t encoderState,
        nvjpegEncoderParams_t encoderParams, cudaStream_t stream) {
    if (data == NULL || jpegSize == 0) return;

    nvjpegStatus_t status;

    // Initialize encoder state
    status = nvjpegEncoderStateCreate(nvjpegHandle, &encoderState, stream);
    if (status != NVJPEG_STATUS_SUCCESS) { 
        return; 
    }

    // Initialize encoder parameters
    status = nvjpegEncoderParamsCreate(nvjpegHandle, &encoderParams, stream);
    if (status != NVJPEG_STATUS_SUCCESS) {
        nvjpegEncoderStateDestroy(encoderState);
        return;
    }

    // Destroy encoder params
    nvjpegEncoderParamsDestroy(encoderParams);

    // Destroy encoder state
    nvjpegEncoderStateDestroy(encoderState);
}

int main(int argc, char* argv[]) {
    const char* filename;
    if (argc > 1) {
        filename = argv[1];
    } else {
        return EXIT_FAILURE;
    }

    nvjpegHandle_t nvjpegHandle;
    nvjpegEncoderState_t encoderState;
    nvjpegEncoderParams_t encoderParams;
    cudaStream_t stream;

    if (nvjpegCreate(NVJPEG_BACKEND_DEFAULT, NULL, &nvjpegHandle) != NVJPEG_STATUS_SUCCESS) {
        return EXIT_FAILURE;
    }

    if (cudaStreamCreate(&stream) != cudaSuccess) {
        nvjpegDestroy(nvjpegHandle);
        return EXIT_FAILURE;
    }

    FILE* inputFile = fopen(filename, "rb");

    //initialization
    unsigned char* jpgData = (unsigned char*)malloc(fileSize);
    size_t bytesRead = fread(jpgData, 1, fileSize, inputFile);
    fclose(inputFile);
    //initialization

    // Process the image
    UT(jpgData, fileSize, nvjpegHandle, encoderState, encoderParams, stream);

    // Cleanup
    free(jpgData);
    cudaStreamDestroy(stream);
    nvjpegDestroy(nvjpegHandle);

    return EXIT_SUCCESS;
}

            The code to be refactored is as follows:
            %s
    		"""


wrap_code_get_var_relationship = """
    You are a CUDA program expert. Read a piece of code, the initialization section includes the initialization of some variables. 
    I want to test the code with different inputs, like fuzz testing.
    You have two tasks: 
    Task 1: Distinguish which variables have independently initialized values(i.e. not controlled by other initialization variables)
    and provide a list just of these independent variable names. 
    If the length of a variable depends on other variables, but the specific values of its elements do not depend on other variables, 
    it is also included in the list. 
    Some fixed type variables should not be considered as mutable variables, such as nvjpegHandle_t nvjpegHandle;
    Task 2: For array type variables, use a 2-dimensions Python list to record the size of the variable. 
    Each element of the list is also a list, corresponding to an array variable. 
    In the second level list, the first element of the list is the name of the array variable, 
    and the following elements are the names of the variables that affect the size of the array.
    
    Format requirements : Just return two python lists. First list is about independent variable names. Second list is about arry variable and its size.
    
    For example, for such an initialization:
    
    //initialization
    int numElements = 256;
    int size = numElements * sizeof(int);
    int *h_data = (int *)malloc(size);
    for (int i = 0; i < numElements; i++) {
        h_data[i] = i;
    }
    int blockSize = 16;
    
    int n = 256;
    double *h_A = (double *)malloc(n*n*sizeof(double))
    for (int i = 0; i < n * n; i++) {
        h_A[i] = (double)i;
    }
    //initialization
                
    the variables that should be written to the  first List is ['numElements', 'h_data', 'blockSize', 'n', 'h_A']. 
    the second list is: [['h_data', 'numElements'], ['h_A', 'n*n*sizeof(double)']].
    Because although the length of h_data is controlled by the variable size, 
    the values of each element in h-data are not controlled by other initialized variables.
    The size variable cannot be written to the List because it is controlled by another initialization variable numElements.
                
    Another example,
    //initialization
    nvjpegHandle_t nvjpegHandle;
    nvjpegEncoderState_t encoderState;
    nvjpegEncoderParams_t encoderParams;
    cudaStream_t stream;
    FILE* inputFile = fopen(filename, "rb");
    unsigned char* jpgData = (unsigned char*)malloc(fileSize);
    size_t bytesRead = fread(jpgData, 1, fileSize, inputFile);
    fclose(inputFile);
    //initialization
                
    the variables that should be written to the first List is ['jpgData']. 
    the second list is: []
    

    The code to be read is as follows:
    %s

    """


if __name__ == "__main__":
    error_messages = """
    20241219_094507.cu(34): error: identifier "cudaMemPoolAccessDesc" is undefined
          cudaMemPoolAccessDesc accessDesc = {0};
          ^
    20241219_094507.cu(35): error: identifier "cudaMemPoolLocationTypeDevice" is undefined
          accessDesc.location.type = cudaMemPoolLocationTypeDevice;
                                     ^
    20241219_094507.cu(37): error: identifier "cudaMemPoolAccessFlagsReadWrite" is undefined
          accessDesc.accessFlags = cudaMemPoolAccessFlagsReadWrite;
    """

    print(get_fake_api(error_messages))