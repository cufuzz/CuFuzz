"""
this .py is for fuzzing a certain cuda harness, and return some information. Before fuzzing, the harness will be
separated into two parts: inputs and function bodies. Then the separated code will be wrapped as a valuable harness.
The wrapper process contains writing input as a .bin file, reading input and linking mutator.o, maintaining the .cur,
and cuda oracle(cuda sig, cuda sanitizer)
"""

import openai
import json
import yaml
from utils_for_test import *
import re
import os
from several_prompt import *
import logging
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)

logger = logging.getLogger()

with open('./config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

openai.api_key = config['llm']['api-key']
openai.base_url = config['llm']['base_url']
openai.default_headers = {"x-foo": "true"}

def harness_inti_separate(code, code_file, compilation, mode, suffix= None) -> bool:
    new_file = f'{code_file[:-3]}{suffix}.cu'
    if mode != 'nvjpeg':                                          ##  for cublas, rt prompt also applies
        code_prompt = harness_separate_code_prompt_4rt%(code)
    elif mode == 'nvjpeg':
        code_prompt = harness_separate_code_prompt_4nvjpeg%(code)

    @retry(
            stop=stop_after_attempt(10),  # 最多重试10次
            wait=wait_exponential(multiplier=1, min=4, max=60),  # 指数退避：初始4秒，最大60秒
            retry=retry_if_exception_type(
                (openai.APIConnectionError,
                 openai.RateLimitError,
                 openai.APITimeoutError,
                 openai.InternalServerError
                 )
            ) # 仅对特定异常重试
        )
    def robust_chat_completion( the_model="gpt-4o"):
        response = openai.chat.completions.create(
            # model="gpt-4o-mini-2024-07-18",
            model=the_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"{code_prompt}",
                        }
                    ]
                },
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "code_description",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": f"A cuda .cu code program",
                            },

                        },
                        "required": [
                            "code",

                        ],
                        "additionalProperties": False
                    }
                }
            },
            temperature=0.5,
            max_tokens=2048,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        return response
    
    response = robust_chat_completion( the_model="gpt-4o")
    content = response.choices[0].message.content
    try:
        data = json.loads(content)
    except Exception as e:
        print(f"error:{e}")
        print(content)

    new_code = data["code"]
    save_harness(new_file, new_code)

    # save harness based on system time
    if compile_code(compilation, suffix=suffix):

        return True
    else:
        return False


def parse_and_wrap_c_file(config:dict, input_filename:str):
    with open(input_filename, 'r') as f:
        the_code = f.read()
    f.close()

    pattern = r"//\s*initialization(.*?)//\s*initialization"
    match = re.search(pattern, the_code, re.DOTALL)
    if match:
        extracted_code = match.group(1).strip() + '\n'
        # print(extracted_code)
    else:
        print("        [!]could not find //initialization part, something wrong? please check _sep harness !!")
        logging.error("        [!]could not find //initialization part, something wrong? please check _sep harness !!")
        return the_code

    ####  determine the varialbes to be mutated using LLM  ###
    code_prompt = wrap_code_get_var_relationship%(extracted_code)

    @retry(
            stop=stop_after_attempt(10),  # 最多重试10次
            wait=wait_exponential(multiplier=1, min=4, max=60),  # 指数退避：初始4秒，最大60秒
            retry=retry_if_exception_type(
                (openai.APIConnectionError,
                 openai.RateLimitError,
                 openai.APITimeoutError,
                 openai.InternalServerError
                 )
            ) # 仅对特定异常重试
        )
    def robust_chat_completion( the_model="gpt-4o"):
        response = openai.chat.completions.create(
            model=the_model,
            # model="gpt-4o-mini-2024-07-18",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"{code_prompt}",
                        }
                    ]
                },
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "variables",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "var1": {
                                "type": "array",
                                "description": f"A list of variables initialized independently.",
                                "items":{
                                    "type":"string"
                                },
                            },
                            "var2": {
                                "type": "array",
                                "description": f"A list of array variables and its size.",
                                "items": {
                                    "type": "array",
                                    "items":{
                                        "type":"string"
                                    }
                                },
                            },
                        },
                        "required": [
                            "var1",
                            "var2",

                        ],
                        "additionalProperties": False
                    }
                }
            },
            temperature=0.9,
            max_tokens=2048,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        return  response

    response = robust_chat_completion( the_model="gpt-4o")
    content = response.choices[0].message.content
    try:
        data = json.loads(content)
    except Exception as e:
        print(f"error:{e}")
        print(content)

    try:
        var_list_independ = data["var1"]
        var_list_array = data["var2"]
    except Exception as e:
        print(f"error:{e}")
        var_list_independ = []
        var_list_array = []

    print(f"        [*] the variables list will be mutated is :{var_list_independ}, | array var size: {var_list_array}")


    ####  set up the buffer and length of the varialbes to be mutated  ###
    wrapped_code = wrap_it(config, the_code, var_list=var_list_independ, var_array=var_list_array)

    # fixed_wrapped_code = check_fix_vulner_of_harness_wrap(wrapped_code)
    if wrapped_code:
        fixed_wrapped_code = wrapped_code
    else:
        fixed_wrapped_code = the_code

    return fixed_wrapped_code


def wrap_it(config, init_code, var_list, var_array):

    loops = config['fuzz_one']['loops']
    norm = config['fuzz_one']['norm']
    lib_name = config['lib']

    ## generate the wrapper for given initialization part of code
    pattern = r"//\s*initialization(.*?)//\s*initialization"
    match = re.search(pattern, init_code, re.DOTALL)
    if match:
        extracted_code = match.group(1).strip() + '\n'
        cleaned_lines = []
        for line in extracted_code.split('\n'):
            if '//' in line:
             # 保留//前的代码部分（若存在）
                code_part = line.split('//')[0]
                if code_part.strip():  # 非空则保留
                    cleaned_lines.append(code_part.rstrip())
            else:
                cleaned_lines.append(line.rstrip())
        extracted_code = '\n'.join(cleaned_lines)
        # print(extracted_code)
    else:
        print("        [!]could not find //initialization part, something wrong? please check _sep harness !!")
        logging.error("        [!]could not find //initialization part, something wrong? please check _sep harness !!")
        return None


    # Initialize an empty list to store the wrap statements
    wrap_prefix = []
    wrap_suffix = []
    wrap_statements = []

    wrap_prefix.append(f"    int loops = 0;")
    wrap_prefix.append(f"    while (loops < {loops}) {{")

    wrap_statements.append(f"\n    // wrap buffer")

    # Regular expression to match local variable types (int, float, etc.)
    local_var_regex = r"\b(int|float|double|char|size_t|unsigned long long|long long|short|unsigned|long)\s+(\w+\s*($$[^$$]*\])*)\s*[^;]*;\n"

    code_lines = extracted_code.splitlines(keepends=True)
    var_name_and_codeline_dict = {}
    for line in code_lines:
        if not line:
            continue  # Skip empty lines

        # Check for local variable declarations
        local_var_match = re.search(local_var_regex, line)
        if local_var_match:
            var_name = local_var_match.group(2).split('[')[0].strip()  # Extract variable name (ignore array brackets)
            if '=' in line:
                init_value = line.split('=', 1)[1]
                var_name_and_codeline_dict[var_name] = init_value
            else:
                var_name_and_codeline_dict[var_name] = None


    # Regular expression to match dynamic memory allocation (malloc)
    malloc_regex = r"(\w+)\s*\*\s*(\w+)\s*=\s*.*(?:malloc|calloc)\((.*)\);"

    # Find all local variables in the code
    local_vars = re.findall(local_var_regex, extracted_code)

    var_array_key = [item[0] for item in var_array]     #  var_array is a 2-level list, get the first element of every sub-list.

    var_type_dict = {} # its format: {var1:'local', var2:'array', var3:'malloc'}

    for var_type, var_name, _ in local_vars:
        if ' ' in var_name:
            var_name = var_name.strip().replace(" ", "")
        if '[' in var_name and ']' in var_name:                                         # that means variable is a array
            var_name, _ = parse_array_declaration(var_name)
            var_type_dict[var_name] = 'array'

        else:
            if var_name in var_list and (var_name not in var_array_key):
                var_type_dict[var_name] = 'local'
            elif var_name in var_list and (var_name in var_array_key):
                var_type_dict[var_name] = 'array'
            else:
                var_type_dict[var_name] = 'copy the definition'

        if var_name in var_list:
            # Wrap local variables (non-malloc)
            if var_type_dict[var_name] == 'local':
                wrap_statements.append(f"    u32 {var_name}_len = sizeof({var_name});")
                wrap_statements.append(f"    u8 *{var_name}_buf = (u8 *)malloc({var_name}_len);")
                wrap_statements.append(f"    memcpy({var_name}_buf, &{var_name}, {var_name}_len);")
                wrap_suffix.append(f"    free({var_name}_buf);")


    # Find all dynamic memory allocations (malloc) in the code
    mallocs = re.findall(malloc_regex, extracted_code)

    for var_type, var_name, size_expr in mallocs:
        var_type_dict[var_name] = 'malloc'
        if var_name in var_array_key :
            if lib_name != 'nvjpeg':
                var_type_dict[var_name] = 'array'
        if (var_name in var_list) and (var_type_dict[var_name] == 'malloc'):
            # Evaluate the malloc size (assuming size_expr is a valid expression like 'numElements * sizeof(int)')
            wrap_statements.append(f"    u32 {var_name}_len = {size_expr};")
            wrap_statements.append(f"    u8 *{var_name}_buf = (u8 *){var_name};")

    wrap_statements.append(f"\n    // wrap havoc and writing .cur")

    for var_type, var_name, size_expr in mallocs:
        if var_name in var_list and (var_type_dict[var_name] == 'malloc'):
            wrap_statements.append(f"    u32 {var_name}_len_havoc;")

    for var_type, var_name, _ in local_vars:
        if ' ' in var_name:
            var_name = var_name.strip().replace(" ", "")
        if var_name in var_list and (var_type_dict[var_name] == 'local'):
            wrap_statements.append(f"    u32 {var_name}_len_havoc;")

    count = 1
    for var_type, var_name, _ in local_vars:

        if ' ' in var_name:
            var_name = var_name.strip().replace(" ", "")

        if var_name in var_list and (var_type_dict[var_name] == 'local'):
            # havoc local variables (non-malloc)
            wrap_statements.append(f"    {var_name}_len_havoc = random_havoc({var_name}_buf, {var_name}_len, 0);")
            
            wrap_statements.append(f"    {var_name} = *({var_type} *){var_name}_buf;")

            if  'double' in var_type or ('float' in var_type):
            ##  For some int variables, modulo 4096
                wrap_statements.append(f"    {var_name} = fmod({var_name}, {norm});")
            else:
                wrap_statements.append(f"    {var_name} %= {norm};")
                wrap_statements.append(f"    if ({var_name}<0) {{ {var_name} = -{var_name} ;}}")
                wrap_statements.append(f"    if ({var_name}==0) {{ {var_name} += 1 ;}}")

            wrap_statements.append(f"    FILE *file_{count} = fopen(\"{count}.bin\", \"wb\");")
            wrap_statements.append(f"    fwrite({var_name}_buf, {var_name}_len, 1, file_{count});")
            wrap_statements.append(f"    fclose(file_{count});")
            count += 1
        # if var_name not in var_list or (var_type_dict[var_name] == 'copy the definition'):
            

    for var_type, var_name, _ in local_vars:
        if '[' in var_name and ']' in var_name:                  # that means variable is a array
            var_name, _ = parse_array_declaration(var_name)
        if ' ' in var_name:
            var_name = var_name.strip().replace(" ", "")

        if var_name in var_list and (var_type_dict[var_name] == 'array'):
            #################
            for array_item in var_array:
                if var_name == array_item[0]:
                    var_name_isarray_size = f'sizeof({var_type})'
                    for i in range(1, len(array_item), 1):
                        var_name_isarray_size += f'* {array_item[i]}'
            #################
            wrap_statements.append(f"    u32 {var_name}_len = {var_name_isarray_size};")
            wrap_statements.append(f"    u8 *{var_name}_buf = (u8 *)malloc({var_name}_len);")
            # wrap_statements.append(f"    memcpy({var_name}_buf, 0, {var_name}_len);")
            wrap_statements.append(f"    u32 {var_name}_len_havoc;")
            wrap_statements.append(f"    {var_name}_len_havoc = random_havoc({var_name}_buf, {var_name}_len, 1);")
            wrap_statements.append(f"    {var_type} *temp_{count} = ({var_type} *)realloc({var_name}, {var_name_isarray_size});")
            wrap_statements.append(f"    {var_name} = temp_{count};")
            wrap_statements.append(f"    memcpy((u8*){var_name}, {var_name}_buf, {var_name}_len);")
            wrap_suffix.append(f"    free({var_name}_buf);")

            wrap_statements.append(f"    FILE *file_{count} = fopen(\"{count}.bin\", \"wb\");")
            wrap_statements.append(f"    fwrite({var_name}_buf, {var_name}_len, 1, file_{count});")
            wrap_statements.append(f"    fclose(file_{count});")
            count += 1
        if var_name not in var_list:
            if lib_name != 'nvjpeg':
                wrap_statements.append(f"    {var_name} = {var_name_and_codeline_dict[var_name]}")
            else:
                if ('read' not in var_name_and_codeline_dict[var_name]) and ('malloc' not in var_name_and_codeline_dict[var_name]):
                    wrap_statements.append(f"    {var_name} = {var_name_and_codeline_dict[var_name]}")

    for var_type, var_name, size_expr in mallocs:
        if var_name in var_list and (var_type_dict[var_name] == 'malloc'):
            if 'jpg' not in var_name and 'image' not in var_name:
                # havoc dynamic variables
                wrap_statements.append(f"    {var_name}_len_havoc = random_havoc({var_name}_buf, {var_name}_len, 1);")

                wrap_statements.append(f"    FILE *file_{count} = fopen(\"{count}.bin\", \"wb\");")
                wrap_statements.append(f"    fwrite({var_name}_buf, {var_name}_len_havoc, 1, file_{count});")
                wrap_statements.append(f"    fclose(file_{count});")
                count += 1
            else:
                ## for jpg image, the first 12 bytes is magic number as file head
                wrap_statements.append(f"    {var_name}_len_havoc = random_havoc({var_name}_buf + 12, {var_name}_len - 12, 1);")

                wrap_statements.append(f"    FILE *file_{count} = fopen(\"{count}.bin\", \"wb\");")
                if 'fileSize' in extracted_code:
                    wrap_statements.append(f"    fwrite({var_name}_buf, fileSize, 1, file_{count});")
                else:
                    wrap_statements.append(f"    fwrite({var_name}_buf, {var_name}_len_havoc, 1, file_{count});")
                wrap_statements.append(f"    fclose(file_{count});")
                count += 1

    for var_type, var_name, size_expr in mallocs:
        if ' ' in var_name:
            var_name = var_name.strip().replace(" ", "")

        if var_name in var_list and (var_type_dict[var_name] == 'array'):
            #################
            # for array_item in var_array:
            #     if var_name == array_item[0]:
            #         var_name_isarray_size = f'sizeof({var_type})'
            #         for i in range(1, len(array_item), 1):
            #             if 'sizeof' not in array_item[i]:
            #                 var_name_isarray_size += f'* {array_item[i]}'
            #################
            if ',' in size_expr:
                var_name_isarray_size = size_expr.replace(',', '*')
            else:
                var_name_isarray_size = size_expr
            wrap_statements.append(f"    u32 {var_name}_len = {var_name_isarray_size};")
            wrap_statements.append(f"    u8 *{var_name}_buf = (u8 *)malloc({var_name}_len);")
            # wrap_statements.append(f"    memcpy({var_name}_buf, 0, {var_name}_len);")
            wrap_statements.append(f"    u32 {var_name}_len_havoc;")
            wrap_statements.append(f"    {var_name}_len_havoc = random_havoc({var_name}_buf, {var_name}_len, 1);")
            wrap_statements.append(f"    {var_type} *temp_{count} = ({var_type} *)realloc({var_name}, {var_name_isarray_size});")
            wrap_statements.append(f"    {var_name} = temp_{count};")
            wrap_statements.append(f"    memcpy((u8*){var_name}, {var_name}_buf, {var_name}_len);")
            wrap_suffix.append(f"    free({var_name}_buf);")

            wrap_statements.append(f"    FILE *file_{count} = fopen(\"{count}.bin\", \"wb\");")
            wrap_statements.append(f"    fwrite({var_name}_buf, {var_name}_len, 1, file_{count});")
            wrap_statements.append(f"    fclose(file_{count});")
            count += 1

    wrap_statements.append(f"\n    // wrap all done!")
    wrap_suffix.append(f"    loops++;}}")


    # Insert the wrap statements into the original C code, just after initialization
    c_code_lines = init_code.splitlines()
    initialization_count = 0
    for i, line in enumerate(c_code_lines):
        if "// initialization" in line or "//initialization" in line or "//  initialization" in line:
            initialization_count += 1
            if initialization_count == 2:
                insert_index = i + 1  # After the second "// initialization"
                # insert_prefix = i + 1

        if "int main(" in line:
            insert_prefix = i + 1


        if "return" in line:
            insert_suffix = i

    # Insert wrap statements after the initialization section
    include_head = [f"#include <fcntl.h>", f"#include <cmath>", f"#include <stdio.h>", f"#include <unistd.h>", f"#include <string.h>", 
                    f'#include "{os.getcwd()}/c_factors/mutate.h"', f'\n']
    wrapped_c_code = include_head + c_code_lines[:insert_prefix] + wrap_prefix + c_code_lines[insert_prefix:insert_index] \
                     + wrap_statements + c_code_lines[insert_index:insert_suffix] + wrap_suffix + c_code_lines[insert_suffix:]

    # Join the lines back into a string
    wrapped_c_code = "\n".join(wrapped_c_code)

    return wrapped_c_code


def check_fix_vulner_of_harness_wrap(the_code: str) -> str:
    """
    the wrapper maybe introduces vulnerabilities, such as mutate a variable too large, and this variable controls the length of array,
    too large length may results overflow. Anther case is that mutate a variable as 0, but this variable will be divided later.
    So check and fix these using llm.
    """

    code_prompt = """
        		You are a CUDA program expert. Read the given code and make modifications to some of the statements. 
        		Mutations have been made to some initialized variables in the code(in the //initialization part). But these mutations may bring some bugs to the code.
        		You need to identify these potential bugs, fix them, and provide me with the complete code after the final fix.
        		You should focus on variables mutated that dynamically allocate memory and variables mutated that act as divisors. 
        		The former may have overflow bugs, while the latter may have non-zero operation bugs.
        		When you discover that a mutated variable is at these two kind of risks, replace it with its initial value.
        		You can only modify the code after '//wrap all done'. Do not change any other code, especially # include

                Format requirements : The code should follow the C or C++ code specification, and the program code should be completed by NVCC and properly formatted.
                In the code, you should write a long sentence without using line breaks, avoiding the newline character \ n.

                Here is a example that you can refer to:

                #include <stdio.h>
                #include <stdlib.h>
                #include <cuda_runtime.h>
                #include <fcntl.h>
                #include <unistd.h>
                #include <string.h>
                #include "mutate.h"

                // define the kernel function
                __global__ void mykernel(int *data, int numElements) {
                    int idx = threadIdx.x + blockIdx.x * blockDim.x;
                    if (idx < numElements) {
                        data[idx] *= 2;
                    }
                }

                int main() {
                    int loops = 0;
                    while (loops < 1000) {
                    // initialization
                    int numElements = 256;
                    int size = numElements * sizeof(int);
                    int *h_data = (int *)malloc(size);
                    for (int i = 0; i < numElements; i++) {
                        h_data[i] = i;
                    }
                    int blockSize = 16;
                        // initialization

                    // wrap buffer
                    u32 numElements_len = sizeof(numElements);
                    u8 *numElements_buf = (u8 *)malloc(numElements_len);
                    memcpy(numElements_buf, &numElements, numElements_len);
                    u32 blockSize_len = sizeof(blockSize);
                    u8 *blockSize_buf = (u8 *)malloc(blockSize_len);
                    memcpy(blockSize_buf, &blockSize, blockSize_len);
                    u32 h_data_len = size;
                    u8 *h_data_buf = (u8 *)h_data;

                    // wrap havoc and writing .cur
                    u32 h_data_len_havoc;
                    u32 numElements_len_havoc = random_havoc(numElements_buf, numElements_len, 0);
                    numElements = *(int *)numElements_buf;
                    FILE *file_1 = fopen("1.bin", "wb");
                    fwrite(numElements_buf, numElements_len, 1, file_1);
                    fclose(file_1);
                    u32 blockSize_len_havoc = random_havoc(blockSize_buf, blockSize_len, 0);
                    blockSize = *(int *)blockSize_buf;
                    FILE *file_2 = fopen("2.bin", "wb");
                    fwrite(blockSize_buf, blockSize_len, 1, file_2);
                    fclose(file_2);
                    h_data_len_havoc = random_havoc(h_data_buf, h_data_len, 1);
                    FILE *file_3 = fopen("3.bin", "wb");
                    fwrite(h_data_buf, h_data_len_havoc, 1, file_3);
                    fclose(file_3);

                    // wrap all done!


                    // allocate memory on the device
                    int *d_data;
                    cudaMalloc((void**)&d_data, size);
                    cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);

                    // Launch the kernel
                    int numBlocks = (numElements + blockSize - 1) / blockSize;
                    mykernel<<<numBlocks, blockSize>>>(d_data, numElements);
                    cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);
                    cudaDeviceSynchronize();

                    // free memory
                    free(h_data);
                    cudaFree(d_data);
                    free(numElements_buf);
                    free(blockSize_buf);
                    loops++;}
                    return EXIT_SUCCESS;
                }

                
                In the example, there are two potential bugs. One is in "mykernel<<<numBlocks, blockSize>>>(d_data, numElements);", 
                because numElements may be mutated to obtain a large number, 
                and the operation of the kernel function may cause overflow of read and write operations on the d_data variable.
                So, fix the line as "mykernel<<<numBlocks, blockSize>>>(d_data, 256);" where 256 is the initialization of numElements.
                Another is in "int numBlocks = (numElements + blockSize - 1) / blockSize;", 
                because blockSize may be mutated to obtain 0, and the bug of zero divided will be triggered.
                So, fix it by add the line like "if (!blockSize){blockSize=16;}"  before "int numBlocks = (numElements + blockSize - 1) / blockSize;". 
                
                the codes you need to modify are as follow:  
                %s
        		""" % (the_code)

    @retry(
            stop=stop_after_attempt(10),  # 最多重试10次
            wait=wait_exponential(multiplier=1, min=4, max=60),  # 指数退避：初始4秒，最大60秒
            retry=retry_if_exception_type(
                (openai.APIConnectionError,
                 openai.RateLimitError,
                 openai.APITimeoutError,
                 openai.InternalServerError
                )
            ) # 仅对特定异常重试
        )
    def robust_chat_completion( the_model="gpt-4o"):
        response = openai.chat.completions.create(
            # model="gpt-4o-mini-2024-07-18",
            model=the_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"{code_prompt}",
                        }
                    ]
                },
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "code_description",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": f"A cuda .cu code program",
                            },

                        },
                        "required": [
                            "code",

                        ],
                        "additionalProperties": False
                    }
                }
            },
            temperature=0.5,
            max_tokens=2048,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        return response
    response = robust_chat_completion( the_model="gpt-4o")
    content = response.choices[0].message.content

    data = json.loads(content)


    fixed_wrapped_code = data["code"]

    return fixed_wrapped_code






if __name__ == "__main__":
    code = """
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Define a device variable to copy data from
__device__ int deviceData = 42;

int main() {
    int hostData;
    size_t size = sizeof(int);

    // Step 1: Copy data from device to host using cudaMemcpyFromSymbol
    cudaError_t err = cudaMemcpyFromSymbol(&hostData, deviceData, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error in cudaMemcpyFromSymbol: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    // Step 2: Check for any errors that occurred in previous calls
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Error after cudaMemcpyFromSymbol: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    // Step 3: Free any allocated resources (none in this case, but included for completeness)
    // Here we would call cudaFree if there were any device memory allocated
    // cudaFree(devicePointer);

    // Step 4: Destroy a stream (not created in this example, but included for completeness)
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaStreamDestroy(stream);

    // Step 5: Destroy a graph (not created in this example, but included for completeness)
    cudaGraph_t graph;
    cudaGraphCreate(&graph, 0);
    cudaGraphDestroy(graph);

    printf("Host data: %s\n", hostData);
    return EXIT_SUCCESS;
}

    """

    # fuzz_one_harness(code, code_file, compilation, suffix= '_sep')

    lib_ = 'curand'
    time_flame = '20250812_215504'
    in_file = f'./{lib_}/harness/{time_flame}/{time_flame}_sep.cu'
    out_file = f'./{lib_}/harness/{time_flame}/{time_flame}_sep_wrap.cu'

    with open('./config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    aa = ['imageData']
    with open(in_file, 'r') as f:
        the_code = f.read()
    f.close()

    cc = parse_and_wrap_c_file(config, input_filename=in_file)
    print(cc)
    save_harness(out_file, cc)

    # with open(in_file, 'r') as f:
    #     content = f.read()
    # f.close()
    # aa = wrap_it(content, ['numElements', 'h_data', 'blockSize'], loops=1000)
    #
    # bb = check_fix_vulner_of_harness_wrap(aa)

    ######      printf("%d\n",loops);


    """   this example for adding CHECK_CUDA
    #include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CHECK_CUDA(call) {\
    cudaError_t err = call;\
    if (err != cudaSuccess) {\
        printf("CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err));\
        exit(1);\
    }\
}

void UT(int device, cudaStream_t &stream, double *h_A, double *d_A, int numElements) {
    // Set the CUDA device
    CHECK_CUDA(cudaSetDevice(device));

    // Create a CUDA stream
    CHECK_CUDA(cudaStreamCreate(&stream));

    // Attach memory to the stream (必须确保d_A已分配且stream已初始化)
    CHECK_CUDA(cudaStreamAttachMemAsync(stream, d_A, 0, cudaMemAttachSingle));
    CHECK_CUDA(cudaStreamSynchronize(stream));  // 同步确保附加完成

    // Initialize cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Perform cuBLAS operation
    double alpha = 1.0, beta = 0.0;
    int incx = 1, incy = 1;
    cublasDgemv(handle, CUBLAS_OP_N, numElements, numElements, &alpha, 
                d_A, numElements, d_A, incx, &beta, d_A, incy);

    cublasDestroy(handle);
}

int main() {
    int device = 0;
    int numElements = 256;
    size_t size = numElements * numElements * sizeof(double);

    // Host memory allocation
    double *h_A = (double *)malloc(size);
    for (int i = 0; i < numElements * numElements; i++) {
        h_A[i] = static_cast<double>(i);
    }

    // Device memory allocation
    double *d_A = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&d_A, size));
    CHECK_CUDA(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));

    // CUDA stream
    cudaStream_t stream;
    UT(device, stream, h_A, d_A, numElements);

    // Cleanup
    CHECK_CUDA(cudaStreamDestroy(stream));
    free(h_A);
    CHECK_CUDA(cudaFree(d_A));

    return EXIT_SUCCESS;
}
    """