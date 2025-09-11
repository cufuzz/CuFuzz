send_request_code_prompt_4rt = """
		Please write a CUDA program with the given CUDA API sequence %s in int main(). The program should meet the following requirements:

        Program Structure: The code should be divided into two parts:
        1. The first part should define the necessary custom kernel function(e.g., __device__, __global__) which will be used by a CUDA API.
        2. The second part should be the int main() function, which calls the CUDA APIs and initializes the arguments of each API. 
        In the main(), if all given CUDA APIs do not call the kernel, the first part could be omitted. 
        Adhere to the given CUDA API sequence and strictly control the order in which APIs appear in the code. 
        There are known API call relationships here: %s
        You should try you best to have the previous API call the next one, according to given call relationships.
        Even if the known call relation is empty, you should try you best to have the previous API call the next one.

        You should pay special attention to the initialization of some variables, especially CUDA variables or structures,
        because incorrect initialization can cause segmentation fault when API calls to these variables in the future.

        Format requirements : The code should follow the C or C++ code specification, and the program code should be complete and properly formatted.
        In the code, you should write a long sentence without using line breaks, avoiding the newline character \ n.
        Try not to use 'printf' in generated code. Don’t make up APIs that don't exist.

        Here is a template function that you can refer to its format, you must specialize the API and initialize the arguments:

        #include <stdio.h>
        #include <stdlib.h>
        #include <cuda_runtime.h>

        // define the kernel function
        __device__ mykernel(){
        ...
        }

        int main() {
        // invoke the given CUDA API according to give order

            arg2 = xxx;
            arg3 = xxx;
            CUDA_API_1(mykernel(), arg2, arg3);

            arg4 = xxx;
            CUDA_API_2(arg4);


            return EXIT_SUCCESS;
        }
		"""
