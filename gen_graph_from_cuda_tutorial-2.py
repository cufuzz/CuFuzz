from typing import Any, Dict, List, Optional, Tuple, Union
import subprocess
import openai
import os
import yaml
import json
import re
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

####   第一版py提取api调用关系，没有提取api调用顺序。而且大量的costomer函数调用cuda api，这种意义不大。
####   因此第二版py修改提示词只提取cuda api的调用，以及一段代码中的api 出现顺序。


the_cublas_split_pages_list = [ [7, 12],[15, 18],[19, 21],[25, 27],[28, 30],[31, 34],[35, 38],[39, 40],[41, 44],[45, 47],[48, 50],[51, 54],
            [55, 59],[60, 63],[64, 66],[67, 71],[72, 76],[77, 81],[82, 87],[88, 90],[91, 93],[95, 98],[99, 104],[105, 110],
            [111, 112],[113, 116],[117, 120],[121, 124],[125, 128],[129, 130],[131, 134],[135, 139],[140, 144],[145,147],
            [148, 151],[152, 156],[157, 161],[162, 166],[167, 170],[171, 174],[175, 180],[181, 183],[184, 187],[198, 200],
            [232, 234],[235, 238],[238, 243],[243, 248],[248, 252],[253, 255],[256, 260],[261, 265],[270, 272],[273, 276],
            [277, 282],[282, 285],[286, 289],[290, 293],[294, 295],[299, 301],[302, 304],[305, 308]
            ]

the_npp_split_pages_list = 'in npp tutorial, there is hardly code example, almost is function signature'

cufft_splited_page = [[9, 10], [23, 26], [85, 88]]

curand_splited_page = [[9, 20], [28, 37], [37,40], [40,47]]

cusolver_splited_page = [[308,310]] 

cusparse_splited_page = [[63,64], [75, 77], [99, 101], [107, 110], [117, 120], [127, 131], [139, 142], [172, 173], [175, 176], 
                         [179, 181], [190, 191], [192, 193]]


with open('./config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

openai.api_key = config['llm']['api-key']
openai.base_url = config['llm']['base_url']

def gpt_output(sys_prompt:str, usr_prompt:str, start_page, end_page, sav_path):
    """
    for pdf tutorial, such as rt-lib, cublas, npp, cufft
    """
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
            "pdf_page": [start_page, end_page],
            "model_response": response_data
        }
        # Write the dictionary as a JSON object to the file
        json.dump(output_dict, json_file)
        json_file.write('\n')


def save_call_and_order_klg_from_pdf(the_pdf_path, the_splited_list):
    """
    call the gpt_output(), generate call and order knowledge from the splited pdf pages.
    this function is appied for rt-lib, cublas, npp, cufft, curand, cusolver, cusparse
    you should change the prompt to determine call or order you tend to get.
    """
    prompts = PROMPTS()
    system_prompt = prompts.prompt_order                                      ##  need change
    
    save_path = './cusparse/pdf2json/gpt4o_output_order.json'                 ##  need change
    
    aaa = the_splited_list
    for i in range(len(aaa)):
        start_page = aaa[i][0] -1
        end_page = aaa[i][1]
        usr_prompt = pdf_to_text_pages(f'./cuda-pdf/{the_pdf_path}', start_page, end_page)
        # glm_output(system_prompt, usr_prompt, start_page, end_page, save_path)
        gpt_output(system_prompt, usr_prompt, start_page, end_page, save_path)
        print(start_page, end_page)


def gpt4o_nvjpeg_output(input_file, sav_path):
    """
    the input_file is a txt file which contains nvjpeg tutorails.
    the generated output is a json. that is the knowledge graph for api ordering and calling
    """
    openai.base_url = "https://api.gpt.ge/v1/"
    openai.default_headers = {"x-foo": "true"}

    prompts = PROMPTS()
    sys_prompt = prompts.prompt_order

    with open(input_file, 'r', encoding='utf-8') as file:
        content = file.read()
    usr_prompt = content

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
            "tutorial_page": os.path.basename(input_file),
            "model_response": response_data
        }
        # Write the dictionary as a JSON object to the file
        json.dump(output_dict, json_file)
        json_file.write('\n')

def gain_nvjpeg_api_sig_dict_by_gpt4o(input_file) -> dict:
    openai.base_url = "https://api.gpt.ge/v1/"
    openai.default_headers = {"x-foo": "true"}

    with open(input_file, 'r', encoding='utf-8') as file:
        content = file.read()
    usr_prompt = content

    sys_prompt = nvjpeg_gain_api_sig_from_txt

    completion = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": usr_prompt},
        ],
    )

    response_content = completion.choices[0].message.content

    try:
        # 移除字符串中的多余字符，以便正确解析JSON
        json_start = response_content.index('```json') + len('```json')
        json_end = response_content.index('```', json_start)
        json_content = response_content[json_start:json_end].strip()
        response_data = json.loads(json_content)

        api_dict = {item['api_name']: item['api_signature'] for item in response_data}
        return api_dict

    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        return


def save_nvjpeg_api_sig_dict():
    in_list = ['nvjpeg_doc_c4_4.2.txt', 'nvjpeg_doc_c4_4.1.txt', 'nvjpeg_doc_c3_3.4.txt', 'nvjpeg_doc_c3_3.3.txt', 'nvjpeg_doc_c3_3.1.txt',
               'nvjpeg_doc_c2_2.3_2.3.5.txt', 'nvjpeg_doc_c2_2.3_2.3.3.txt', 'nvjpeg_doc_c2_2.3_2.3.2.txt', 'nvjpeg_doc_c2_2.3_2.3.1.txt',
               'nvjpeg_doc_c2_2.1.txt']

    all_dict = {}
    for item in in_list:
        in_file = f'./nvjpeg/split_by_2_3level_heading/{item}'
        the_dict = gain_nvjpeg_api_sig_dict_by_gpt4o(in_file)
        all_dict.update(the_dict)
        print(the_dict)

    with open('./nvjpeg/tutorial2json/nvjpeg_api_sig_dict.pkl', 'wb') as f:
        pickle.dump(all_dict, f)

    for k, v in all_dict.items():
        print(k,v)


def gain_cublas_cunpp_api_sig_dict_by_gpt4o(input_file) -> dict:
    openai.base_url = "https://api.gpt.ge/v1/"
    openai.default_headers = {"x-foo": "true"}

    usr_prompt = input_file

    sys_prompt = cublas_cunpp_gain_api_sig_from_pdf_seg

    completion = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": usr_prompt},
        ],
    )

    response_content = completion.choices[0].message.content

    try:
        # 移除字符串中的多余字符，以便正确解析JSON
        json_start = response_content.index('```json') + len('```json')
        json_end = response_content.index('```', json_start)
        json_content = response_content[json_start:json_end].strip()
        response_data = json.loads(json_content)

        api_dict = {item['api_name']: item['api_signature'] for item in response_data}
        return api_dict

    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        return

def save_cublas_cufft_curand_cusolver_cusparse_api_sig_dict(the_splite_page_list=the_cublas_split_pages_list, 
                                                            the_pdf_path='./cuda-pdf/CUBLAS_Library.pdf',
                                                            the_save_path='./cublas/pdf2json/cublas_api_sig_dict.pkl'):
    
    all_dict = {}

    aaa = the_splite_page_list

    for i in range(len(aaa)):
        start_page = aaa[i][0] - 1
        end_page = aaa[i][1]
        usr_prompt = pdf_to_text_pages(the_pdf_path, start_page, end_page)
        the_dict = gain_cublas_cunpp_api_sig_dict_by_gpt4o(usr_prompt)
        all_dict.update(the_dict)
        print(the_dict)
        print(start_page, end_page)

    with open(the_save_path, 'wb') as f:
        pickle.dump(all_dict, f)

    for k, v in all_dict.items():
        print(k, v)


def save_cunpp_api_sig_dict():    
    count = 0
    the_api_names = set()
    json_path = f'./cunpp/cudasample2json/gpt4o_output_calls.json'
    with open(json_path, 'r', encoding='utf-8') as file:
        for line in file:
            count += 1
            try:
                # 尝试解析每一行为JSON对象
                data = json.loads(line)
                # 打印读取到的数据
                
                for item in data['model_response']:
                    if ('cu' in item['head']) or ('CU' in item['head']) or ('npp' in item['head']) or ('NPP' in item['head']):
                        the_api_names.add(item['head'])
                    if ('cu' in item['tail']) or ('CU' in item['tail']) or ('npp' in item['tail']) or ('NPP' in item['tail']):
                        the_api_names.add(item['tail'])

            except json.JSONDecodeError as e:
                # 如果解析出错，打印错误信息
                print(f"Error parsing JSON line: {e}")
    print(len(the_api_names))

    json_path = f'./cunpp/cudasample2json/gpt4o_output_order.json'
    with open(json_path, 'r', encoding='utf-8') as file:
        for line in file:
            count += 1
            try:
                # 尝试解析每一行为JSON对象
                data = json.loads(line)
                # 打印读取到的数据
                for item in data['model_response']['order']:
                    for item1 in item:
                        if ('cu' in item1) or ('CU' in item1) or ('npp' in item1) or ('NPP' in item1):
                            the_api_names.add(item1)

            except json.JSONDecodeError as e:
                # 如果解析出错，打印错误信息
                print(f"Error parsing JSON line: {e}")
    print(len(the_api_names))

    print(the_api_names)

    all_dict = {}

    bbb = {'nppGetLibVersion':[50, 55], 'nppiLabelMarkersUF_8u32u_C1R_Ctx':[1950,1955], 'nppiLabelMarkersUFGetBufferSize_32u_C1R':[1950,1955],
           'nppiLabelMarkersUFBatch_8u32u_C1R_Advanced_Ctx':[1955,1960], 'nppiCompressMarkerLabelsGetBufferSize_32u_C1R':[1960,1965],
           'nppiCompressMarkerLabelsUF_32u_C1IR':[1960,1965], 'nppiCompressMarkerLabelsUF_32u_C1IR_Ctx':[1960,1965],
           'nppiSegmentWatershed_8u_C1IR_Ctx':[1965,1970], 'nppiSegmentWatershedGetBufferSize_8u_C1R':[1965,1970]}
    
    for  v in [[50,55], [1950,1955], [1955,1960], [1960,1965], [1965,1970]]:
        start_page = v[0] 
        end_page = v[1]
        usr_prompt = pdf_to_text_pages('./cuda-pdf/NPP_Library.pdf', start_page, end_page)
        the_dict = gain_cublas_cunpp_api_sig_dict_by_gpt4o(usr_prompt)
        all_dict.update(the_dict)

    # aaa = [ [i, i+5] for i in range(0, 3289, 5)]
    # for i in range(len(aaa)):
    #     start_page = aaa[i][0] 
    #     end_page = aaa[i][1]
    #     usr_prompt = pdf_to_text_pages('./cuda-pdf/NPP_Library.pdf', start_page, end_page)
    #     button = False

    #     ccc = ''
    #     for api_name in the_api_names:
    #         if api_name in usr_prompt:
    #             button = True
    #             ccc = api_name
    #             print(f'page {start_page} ~ page {end_page} | api name {api_name}')
    
    #     if button:
    #         print(f'page {start_page} ~ page {end_page} | api name {api_name}')

    #         the_dict = gain_cublas_cunpp_api_sig_dict_by_gpt4o(usr_prompt)
    #         all_dict.update(the_dict)
    #         print(the_dict)
    #         print(start_page, end_page)

    with open('./cunpp/pdf2json/cunpp_api_sig_dict.pkl', 'wb') as f:
        pickle.dump(all_dict, f)

    for k, v in all_dict.items():
        print(k, v)


def glm_output(sys_prompt:str, usr_prompt:str, start_page, end_page, sav_path):
    # final_output = ""

    response = client.chat.completions.create(
        model="glm-4",  # 请填写要调用的模型名称
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": usr_prompt},
        ],
        # stream=True,
    )

    # for chunk in response:
    #     print(chunk.choices[0].delta.content, end="", flush=True)
    #     final_output += chunk.choices[0].delta.content

    response_content = response.choices[0].message.content
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
            "pdf_page": [start_page, end_page],
            "model_response": response_data
        }
        # Write the dictionary as a JSON object to the file
        json.dump(output_dict, json_file)
        json_file.write('\n')




if __name__ == "__main__":   ### 序号为pdf的[开始页数-1，结束页数]
    # start_page = 270
    # end_page = 274
    # usr_prompt = pdf_to_text_pages('/home/fanximing/cuda-pdf/CUDA_C_Programming_Guide.pdf', start_page, end_page)
    # glm_output(system_prompt, usr_prompt, start_page, end_page)
    # print(start_page, end_page)

    ###  每次输入3页左右，太长的问题，提取质量下降，太短，上下文描述不完整。  for cuda rt lib.
    ###  已经记录过的页码: [26,29] [29, 32] [32, 35] [41,45] [45,49] [49,54] [55,58] [60,62] [63,67] [67,70] [72,75] [75,77]
    ###  [77,81] [81,87] [87,91] [91,95] [95,98] [98,101] [101,105] [106,108] [108,111] [111,114] [113,116] [115,118]
    ###  [118,122] [122,125] [125,127] [127,130] [129,133] [133,136] [135,139] [138, 142] [142,145] [145,148] [157,160]
    ###  [185,188] [188,189] [201,206] [207,210] [209,213] [212,215] [216,218] [227,230] [230, 232] [233,235] [236,239]
    ###  [241,243] [243,246] [246,249] [254,259] [259,261] [264,267] [267,270] [270,274] [281,284]
    ###  [284,287] [287,290] [290,293] [293,296] [296, 299] [299,303] [303, 307], [319, 322], [323,327]
    ###  [334,337], [341,344], [345,348], [348,352], [355,357], [357,360], [360, 363], [363,365], [365,369], [368,372],\
    ###  [375,379], [379, 382], [386,388],[483,485], [485, 487], [487,490], [490, 493], [493,496], [496,499], \
    ###  [511,514], [514,518], [518,521], [521, 525], [527,529], [532,535], [547,550]


    ### for gpt output, cuda runtime api, txt to json, establish knowledge graph
    # aaa = [
    #
    #          [284,287], [287,290], [290,293] ,[293,296], [296, 299], [299,303] ,[303, 307], [319, 322], [323,327]
    #         , [334,337], [341,344], [345,348], [348,352], [355,357], [357,360], [360, 363], [363,365], [365,369], [368,372]
    #         , [375,379], [379, 382], [386,388],[483,485], [485, 487], [487,490], [490, 493], [493,496], [496,499]
    #         , [511,514], [514,518], [518,521], [521, 525], [527,529], [532,535], [547,550]]
    #
    #
    # ccc = [[299,303]]
    #
    # prompts = PROMPTS()
    # system_prompt = prompts.prompt_order
    #
    # save_path = './rt-lib/pdf2json/gpt4o_output_order.json'
    #
    # for i in range(len(aaa)):
    #     start_page = aaa[i][0] -1
    #     end_page = aaa[i][1]
    #     usr_prompt = pdf_to_text_pages('./cuda-pdf/CUDA_C_Programming_Guide.pdf', start_page, end_page)
    #     # glm_output(system_prompt, usr_prompt, start_page, end_page, save_path)
    #     gpt_output(system_prompt, usr_prompt, start_page, end_page, save_path)
    #     print(start_page, end_page)
    ### for gpt output, cuda runtime api, txt to json, establish knowledge graph


    # aaa = cusparse_splited_page                                                           ##  need change
    # save_call_and_order_klg_from_pdf('CUSPARSE_Library.pdf', aaa)                         ##  need change


    aaa = [[42,48],[49,49],[52,59],[60,62],[64,66],[68,71],[73,74],[77,86],[88,89],[91,92],[94,98],[102,106],[110,112],[113,114],[116,116],
           [120,127],[131,133],[135,138],[142,145],[147,152],[154,158],[160,160],[163,171],[174,179],[182,184],[186,188],[190,192],[194,196],
           [198,200],[202,204],[207,209],[211,213],[216,216],[221,217],[228,235],[236,241],[242,252],[254,256],[258,261],[264,265],[268,268],
           [274,275],[278,278],[281,282],[286,287],[291,292],[294,296]]
    read_path = './cuda-pdf/CUSPARSE_Library.pdf'
    save_path = './cusparse/pdf2json/cusparse_api_sig_dict.pkl'
    save_cublas_cufft_curand_cusolver_cusparse_api_sig_dict(aaa, read_path, save_path)
