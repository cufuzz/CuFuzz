from typing import Any, Dict, List, Optional, Tuple, Union
import yaml
import openai
import os
import json
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
    TextStreamer
)
from accelerate import infer_auto_device_map, Accelerator

from zhipuai import ZhipuAI
from pdf2text import *
from several_prompt import *
from pdf2text import *


client = ZhipuAI(api_key="6c88acb666fbc9cab823761a9aab62f8.vUJGUaydVzyBwJEG")  # 请填写自己的APIKey
with open('./config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

openai.api_key = config['llm']['api-key']
openai.base_url = config['llm']['base_url']

def gpt_output_for_single_cu(sys_prompt:str, usr_prompt:str, cu_path, sav_path):
    openai.base_url = "https://api.gpt.ge/v1/"
    openai.default_headers = {"x-foo": "true"}

    completion = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": usr_prompt},
        ],
    )

    response_content = completion.choices[0].message.content
    print(response_content)

    try:
        # 移除字符串中的多余字符，以便正确解析JSON
        json_start = response_content.index('```json') + len('```json')
        json_end = response_content.index('```', json_start)
        json_content = response_content[json_start:json_end].strip()
        response_data = json.loads(json_content)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        return

    ###  for stream output
    with open(sav_path, 'a') as json_file:
        # Convert the chunk to a dictionary and save it as JSON

        output_dict = {
            "cu_path": cu_path,
            "model_response": response_data
        }
        # Write the dictionary as a JSON object to the file
        json.dump(output_dict, json_file)
        json_file.write('\n')

    print(cu_path + "**** has been saved!")


def check111(path):
    file_list = os.listdir(path)
    ccc = 0
    for item in file_list:
        if ('.cu' in item) and ('.cuh' not in item):
            ccc += 1
            if ccc >1:
                print(os.path.join(path, item))


def get_graph_for_diff_format_from_sample_file(cuda_sample_path:str, system_prompt:str, mode:str = ' ') -> List:
    """ 从cuda sample中找到所有的 .cu  .cpp 示例文件, 针对不同的文件形式进行分类，对不同类别采用不同的提示词工程 """
    count = 0
    count1, count2, count3, count4, count5, count6, count7, count0 = 0, 0, 0, 0, 0, 0, 0, 0
    cccc = 0

    sub_list = os.listdir(cuda_sample_path)
    for s1 in sub_list:
        ss_list = os.listdir(os.path.join(cuda_sample_path, s1))
        for s2 in ss_list:
            s3_file = os.path.join(cuda_sample_path, s1, s2)
            if os.path.isdir(s3_file):
                # print(os.path.join(cuda_sample_path, s1, s2))
                count += 1

                """ 判断示例代码的文件结构, 有以下几种形式
                           cu,  main.cpp,   xx.cpp
                1. count1:  0，        0,      1
                2. count2:  0，        1,      0
                3. count3:  0，        1,      1
                4. count4:  1，        0,      0            only .cu          88 cases
                5. count5:  1，        0,      1            .cu and xx.cpp    68 cases
                6. count6:  1，        1,      0
                7. count7:  1，        1,      1
                0. count1:  0，        0,      0
                """
                sss_list = os.listdir(s3_file)
                button1, button2, button3 = False, False, False
                for s4_file in sss_list:
                    if '.cu' in s4_file:
                        button1 = True
                    if 'main.cpp' in s4_file:
                        button2 = True
                    if ('main.cpp' not in s4_file) and '.cpp' in s4_file:
                        button3 = True

                if (not button1) and (not button2) and (button3):
                    count1 += 1

                if (not button1) and (button2) and (not button3):
                    count2 += 1

                if (not button1) and ( button2) and (button3):
                    count3 += 1

                if (button1) and (not button2) and (not button3):
                    count4 += 1
                    if 'only_cu' in mode:
                        for s4_file in sss_list:
                            if ('.cu' in s4_file) and ('.cuh' not in s4_file):
                                cu_full_path = os.path.join(s3_file, s4_file)
                                cu_context = read_cu_file(cu_full_path)
                                save_path = './rt-lib/cudasample2json/gpt4o_output_calls_only_cu.json'
                                # print(cu_context)
                                gpt_output_for_single_cu(system_prompt, cu_context, cu_full_path, save_path)


                if ( button1) and (not button2) and (button3):
                    count5 += 1
                if (button1) and ( button2) and (not button3):
                    count6 += 1
                if ( button1) and ( button2) and (button3):
                    count7 += 1
                if ( not button1) and (not button2) and (not button3):
                    count0 += 1
                    # print(s3_file)


    # print("in cuda-sample, || total count: %s || only .cu: %s || only main.cpp: %s || .cu and mian: %s || no cu no main just cpp: %s"% \
    #       (count, only_cu_conut, only_maincpp_count, cu_maincpp_count, only_cpp_exp_main_count))
    print("count1: %s||count2: %s||count3: %s||count4: %s||count5: %s||count6: %s||count7: %s|| count0: %s|| total: %s"%\
          (count1, count2, count3, count4, count5, count6, count7, count0, count))


def gpt_output_for_nvjpeg_blas_npp(sys_prompt:str, usr_prompt:str, in_path, sav_path):
    openai.base_url = "https://api.gpt.ge/v1/"
    openai.default_headers = {"x-foo": "true"}

    completion = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": usr_prompt},
        ],
    )

    response_content = completion.choices[0].message.content
    print(response_content)

    try:
        # 移除字符串中的多余字符，以便正确解析JSON
        json_start = response_content.index('```json') + len('```json')
        json_end = response_content.index('```', json_start)
        json_content = response_content[json_start:json_end].strip()
        response_data = json.loads(json_content)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        return

    ###  for stream output
    with open(sav_path, 'a') as json_file:
        # Convert the chunk to a dictionary and save it as JSON

        output_dict = {
            "cu_path": in_path,
            "model_response": response_data
        }
        # Write the dictionary as a JSON object to the file
        json.dump(output_dict, json_file)
        json_file.write('\n')

    print(in_path + "**** has been saved!")


def gen_knowledge_from_cu_aimto_cublas():

    cuda_sample_path = '../CUDALibrarySamples/cuBLAS'
    prompts = PROMPTS()
    system_prompt = prompts.prompt_order_from_cuda_sample_only_cu

    h_file = f'{cuda_sample_path}/utils/cublas_utils.h'

    for f_1 in ['Extensions', 'Level-1', 'Level-2', 'Level-3']:
        for f_2 in os.listdir(os.path.join(cuda_sample_path, f_1)):
            for f_3 in os.listdir(os.path.join(cuda_sample_path, f_1, f_2)):
                if '.cu' in f_3:

                    cpp_file = os.path.join(cuda_sample_path, f_1, f_2, f_3)
                    print(cpp_file)
                    with open(h_file, 'r', encoding='utf-8') as file:
                        content_h = file.read()

                    usr_prompt = "The given code consists of .h file and .cpp file. The .h codes are as follow:\n\n" + content_h

                    with open(cpp_file, 'r', encoding='utf-8') as file:
                        content_cpp = file.read()

                    usr_prompt += "\nThe corresponding .cpp codes are as follow:\n\n" + content_cpp

                    save_path = './cublas/cudasample2json/gpt4o_output_order.json'

                    gpt_output_for_nvjpeg_blas_npp(system_prompt, usr_prompt, cpp_file, save_path)



def gen_knowledge_from_cu_aimto_nvjpeg():
    #####  for nvjpeg library
    cuda_sample_path = '../CUDALibrarySamples/nvJPEG/nvJPEG-Decoder-Backend-ROI'
    prompts = PROMPTS()
    system_prompt = prompts.prompt_call_from_cuda_sample_only_cu

    h_file = f'{cuda_sample_path}/nvJPEGROIDecode.h'
    cpp_file = f'{cuda_sample_path}/nvJPEGROIDecode.cpp'

    with open(h_file, 'r', encoding='utf-8') as file:
        content_h = file.read()

    usr_prompt = "The given code consists of .h file and .cpp file. The .h codes are as follow:\n\n" + content_h

    with open(cpp_file, 'r', encoding='utf-8') as file:
        content_cpp = file.read()

    usr_prompt += "\nThe corresponding .cpp codes are as follow:\n\n" + content_cpp

    save_path = './nvjpeg/sample2json/gpt4o_output_calls.json'

    gpt_output_for_nvjpeg_blas_npp(system_prompt, usr_prompt, cpp_file, save_path)


def gen_knowledge_from_cu_aimto_cunpp():
    #####  for cuNPP library

    for item in ['batchedLabelMarkersAndCompression', 'distanceTransform', 'findContour', 'watershedSegmentation']:
        cuda_sample_path = f'../CUDALibrarySamples/NPP/{item}'
        prompts = PROMPTS()
        system_prompt = prompts.prompt_order_from_cuda_sample_only_cu

        h_file = []
        cpp_file = []    
        for cpp_h_file in os.listdir(cuda_sample_path):
            if '.cpp' in cpp_h_file:
                cpp_file.append(f'{cuda_sample_path}/{cpp_h_file}')
            if '.h' in cpp_h_file:
                h_file.append(f'{cuda_sample_path}/{cpp_h_file}')

        print(h_file)
        print(cpp_file)

        content_h = ""
        for one_h_file in h_file:
            with open(one_h_file, 'r', encoding='utf-8') as file:
                content_h += file.read()
                content_h += "\n"

        usr_prompt = "The given code consists of .h file and .cpp file. The .h codes are as follow:\n\n" + content_h

        with open(cpp_file[0], 'r', encoding='utf-8') as file:
            content_cpp = file.read()

        usr_prompt += "\nThe corresponding .cpp codes are as follow:\n\n" + content_cpp

        save_path = './cunpp/cudasample2json/gpt4o_output_order.json'

        gpt_output_for_nvjpeg_blas_npp(system_prompt, usr_prompt, cpp_file[0], save_path)


def gen_knowledge_from_cu_aimto_cufft_curand_cusolver(lib_name, mode='call'):
    #####  for cuFFT, cuRAND, cuSOLVER library

    lib_path = f'../CUDALibrarySamples/{lib_name}'
    dir_list = []
    if lib_name == 'cuFFT':
        dir_list = [
            item for item in os.listdir(lib_path) 
            if os.path.isdir(os.path.join(lib_path, item)) and 'util' not in item and 'lto' not in item
        ]
    elif lib_name == 'cuRAND':
        cc = os.listdir(lib_path+'/Host') 
        dir_list = [
            item for item in os.listdir(lib_path +'/Host') 
            if os.path.isdir(os.path.join(lib_path +'/Host', item)) 
        ]

    elif lib_name == 'cuSOLVER':
        dir_list = [
            item for item in os.listdir(lib_path) 
            if os.path.isdir(os.path.join(lib_path, item)) and 'util' not in item and 'cmake' not in item
        ]


    for item in dir_list:
        h_file = []
        cpp_file = []  

        cuda_sample_path = f'../CUDALibrarySamples/{lib_name}/{item}'
        if lib_name == 'cuRAND':
            cuda_sample_path = f'../CUDALibrarySamples/{lib_name}/Host/{item}'

        prompts = PROMPTS()

        if mode == 'call':
            system_prompt = prompts.prompt_call_from_cuda_sample_only_cu
        elif mode == 'order':
            system_prompt = prompts.prompt_order_from_cuda_sample_only_cu

          
        for cpp_h_file in os.listdir(cuda_sample_path):
            if ('.cpp' in cpp_h_file) or ('.cu' in cpp_h_file):
                cpp_file.append(f'{cuda_sample_path}/{cpp_h_file}')


        for iitem in os.listdir(f'../CUDALibrarySamples/{lib_name}/utils'):
            if '.h' in iitem:
                h_file.append(f'../CUDALibrarySamples/{lib_name}/utils/{iitem}')

        print(h_file)
        print(cpp_file)

        content_h = ""
        for one_h_file in h_file:
            with open(one_h_file, 'r', encoding='utf-8') as file:
                content_h += file.read()
                content_h += "\n"

        usr_prompt = "The given code consists of .h file and .cpp file. The .h codes are as follow:\n\n" + content_h

        for one_cpp in cpp_file:
            with open(one_cpp, 'r', encoding='utf-8') as file:
                content_cpp = file.read()

            usr_prompt += "\nThe corresponding .cpp or .cu codes are as follow:\n\n" + content_cpp

            save_path = f'./{lib_name.lower()}/cudasample2json/gpt4o_output_calls.json'
            if mode == 'order':
                save_path = f'./{lib_name.lower()}/cudasample2json/gpt4o_output_order.json'

            gpt_output_for_nvjpeg_blas_npp(system_prompt, usr_prompt, one_cpp, save_path)


def gen_knowledge_from_cu_aimto_cusparse(lib_name='cuSPARSE', mode='call'):
    #####  for cuSPARSE library

    lib_path = f'../CUDALibrarySamples/cuSPARSE'
    
    dir_list = [
        item for item in os.listdir(lib_path) 
        if os.path.isdir(os.path.join(lib_path, item)) 
    ]
    
    for item in dir_list:
        h_file = []
        cpp_file = []  

        cuda_sample_path = f'../CUDALibrarySamples/{lib_name}/{item}'
        
        prompts = PROMPTS()

        if mode == 'call':
            system_prompt = prompts.prompt_call_from_cuda_sample_only_cu
        elif mode == 'order':
            system_prompt = prompts.prompt_order_from_cuda_sample_only_cu
         
        for cpp_h_file in os.listdir(cuda_sample_path):
            if ('.cpp' in cpp_h_file) or ('.cu' in cpp_h_file) or ('.c' in cpp_h_file):
                cpp_file.append(f'{cuda_sample_path}/{cpp_h_file}')
            
        print(h_file)
        print(cpp_file)

        usr_prompt = "The given codes are as follow:\n\n"

        for one_cpp in cpp_file:
            with open(one_cpp, 'r', encoding='utf-8') as file:
                content_cpp = file.read()

            usr_prompt += "\nThe corresponding .cpp or .cu codes are as follow:\n\n" + content_cpp

            save_path = f'./{lib_name.lower()}/cudasample2json/gpt4o_output_calls.json'
            if mode == 'order':
                save_path = f'./{lib_name.lower()}/cudasample2json/gpt4o_output_order.json'

            gpt_output_for_nvjpeg_blas_npp(system_prompt, usr_prompt, one_cpp, save_path)



if __name__ == "__main__":   ### 序号为pdf的[开始页数-1，结束页数]
    # start_page = 270
    # end_page = 274
    # usr_prompt = pdf_to_text_pages('/home/fanximing/cuda-pdf/CUDA_C_Programming_Guide.pdf', start_page, end_page)
    # glm_output(system_prompt, usr_prompt, start_page, end_page)

    #####  for cuda rt library
    # cuda_sample_path = '/home/fanximing/cuda-samples/Samples'
    # prompts = PROMPTS()
    # system_prompt = prompts.prompt_call_from_cuda_sample_only_cu
    #
    # mode =  'gen_from_only_cu'
    #
    # get_graph_for_diff_format_from_sample_file(cuda_sample_path, system_prompt, mode)
    #####  for cuda rt library

    
    ttt = 'cuSPARSE'
    # gen_knowledge_from_cu_aimto_cusparse(ttt, 'call')
    gen_knowledge_from_cu_aimto_cusparse(ttt, 'order')

