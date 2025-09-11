import json
import os
import datetime
import yaml
import subprocess
from Extend_Grapy import *
from several_prompt import *
from establish_graph_from_json import *
from fuzz_exec_para import *
from utils_for_test import *
from oracle import celibrated_harness, get_top_indices, get_one_harness_candi_mutate
from fuzz_one import *
from mutator import *
import time
import sys
import logging
import openai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)

with open('./config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

openai.api_key = config['llm']['api-key']
openai.base_url = config['llm']['base_url']
openai.default_headers = {"x-foo": "true"}

logging.basicConfig(filename='log.txt', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Chatgpt:
    def __init__(self, **kwargs):
        self.code = None
        self.code_file = None
        self.compilation = None
        self.compile_result = None
        self.folder_name = None   ##  this is the {timetemp}, responding to a harness, we put all kinds of elements in it.

        if 'prompt' in kwargs:
            self.prompt = kwargs['prompt']

        if 'analysis' in kwargs:
            self.analysis = kwargs['analysis']

    def extract_api_calls_and_order(self):
        """
        this function is used to extract the relationship of api calls and order for the generated code, when the code
        has been compiled.
        """

    def send_request(self, the_sequence, the_call_edge, mode):
        signature_list = search_signature(the_sequence, mode)

        if mode == 'rt':
            code_prompt = send_request_code_prompt_4rt%(the_sequence, the_call_edge, signature_list)
        elif mode == 'nvjpeg':
            code_prompt = send_request_code_prompt_4nvjpeg%(the_sequence, the_call_edge, signature_list)
        elif mode == 'cublas':                                                    ##  for cublas, rt prompt also applies
            code_prompt = send_request_code_prompt_4cublas % (the_sequence, the_call_edge, signature_list)
        elif mode == 'cunpp':
            code_prompt = send_request_code_prompt_4nvjpeg%(the_sequence, the_call_edge, signature_list)
        elif mode == 'cufft' or (mode == 'curand') or (mode == 'cusolver'):
            code_prompt = send_request_code_prompt_4cufft % (the_sequence, the_call_edge,signature_list)


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
        def robust_chat_completion(the_messages, the_model="gpt-4o"):
            response = openai.chat.completions.create(
                # model="gpt-4o-mini-2024-07-18",
                model=the_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": the_messages,
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
                                "compiler": {
                                    "type": "string",
                                    "description": "The compilation command for the generated code. Using a.cu to refer to the code, and a.out to execution file. If a dynamic link library is needed, add it to the compilation command."
                                },
                            },
                            "required": [
                                "code",
                                "compiler",
                            ],
                            "additionalProperties": False
                        }
                    }
                },
                temperature=0.4,
                max_tokens=2048,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )
            return response

        response = robust_chat_completion(the_messages=code_prompt, the_model="gpt-4o")

        content = response.choices[0].message.content
        try:
            data = json.loads(content)
        except Exception as e:
            print(f"error:{e}")
            print(content)

        self.code = data["code"]
        compilation = data["compiler"]
        self.compilation = compilation

        # save harness based on system time
        now = datetime.datetime.now()
        timestemp = now.strftime("%Y%m%d_%H%M%S")

        try:
            if mode ==  'rt':
                os.mkdir(f'./rt-lib/harness/{timestemp}')
            elif mode == 'nvjpeg':
                os.mkdir(f'./nvjpeg/harness/{timestemp}')
            elif mode == 'cublas':
                os.makedirs(f'./cublas/harness/{timestemp}')
            else:
                os.makedirs(f'./{mode}/harness/{timestemp}')
            self.folder_name = timestemp
        except Exception as e:
            print(f"{timestemp} mkdir failed. Process interrupted")
            sys.exit(1)

        if mode == 'rt':
            self.code_file = f'./rt-lib/harness/{timestemp}/{timestemp}.cu'

            self.compilation = compilation.replace(' a.cu', f' ./rt-lib/harness/{timestemp}/{timestemp}.cu').replace(' a.out', f' ./rt-lib/harness/{timestemp}/{timestemp}').replace(\
                'nvcc', '/usr/local/cuda/bin/nvcc')
        else:
            self.code_file = f'./{mode}/harness/{timestemp}/{timestemp}.cu'

            self.compilation = compilation.replace(' a.cu', f' ./{mode}/harness/{timestemp}/{timestemp}.cu').replace(' a.out', f' ./{mode}/harness/{timestemp}/{timestemp}').replace(\
                'nvcc', '/usr/local/cuda/bin/nvcc')


        if mode in ['nvjpeg', 'cublas', 'cufft', 'curand', 'cusolver', 'cusparse']:
            if f'-l{mode}' not in self.compilation:
                self.compilation += f" -l{mode}"
        # if mode == 'cublas':
        #     if '-lcublas' not in self.compilation:
        #         self.compilation += " -lcublas"
        


        save_harness(self.code_file, self.code)

        return self.code, self.compilation

    def generate_input_data(self):
        self.prompt = 'read the code and generate the input data which match the format according to the code'
        response = openai.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",
            # model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"{self.prompt}:{self.code}",
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
                            "data": {
                                "type": "string",
                                "description": "A data file for AFL(American Fuzzy Loop) to fuzz the code you just generated.This data can be saved as a data file and directly given to AFL as input. Ensure there are no extra symbols. DO not use ellipsis"
                            },
                        },
                        "required": [
                            "data",
                        ],
                        "additionalProperties": False
                    }
                }
            },
            temperature=0.7,
            max_tokens=2048,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        content = response.choices[0].message.content
        try:
            data = json.loads(content)
        except Exception as e:
            print(e)
        input_data_2 = data["data"]
        return input_data_2

    # this function is to optimize the code that gpt just generate
    def optimizer(self):
        raise NotImplementedError

    # this function is to fix the code that gpt just generate
    # if there is a compilation error, the gpt will fix the code based on the error message
    # Moreover, we set some error type, like undefined, if these error types appears in message, it will render gpt add some related .h.
    def compile_code(self):
        compile_command = shlex.split(self.compilation)
        try:
            result = subprocess.run(compile_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                    check=True)
            print(f"   Compilation successful: {compile_command}")
            self.compile_result = result.stdout + '\n' + result.stderr
            return True
        except subprocess.CalledProcessError as e:
            print(f"   Compilation failed: {compile_command}")
            result = e.stderr
            self.compile_result = result
            return False

    def bugfix(self, mode):
        fake_api_list = []
        error_argues_api_dic = {}
        if ('is undefined' or 'argument') in self.compile_result:
            ## that means some of apis are most likely made up by llm. find it and delete it
            fake_api_list = get_fake_api(self.compile_result)
            error_argues_api_dic, _ = get_arguement_fix(self.compile_result, self.code_file, mode)

        suggestions = f'Delete the api: {fake_api_list}'
        suggestions_2 = ' '
        if error_argues_api_dic:
            suggestions_2 += f'Modify some API arguments, the correct API signatures are: '
            for k,v in error_argues_api_dic.items():
                if len(v) > 1:
                    suggestions_2 += f'{v}, '
                else:
                    suggestions_2 += f'{v[0]}, '

        self.prompt = """
Read the code and the corresponding compilation error message, then repair the code to make it compile correctly. 
Please adhere to the repair suggestions that have been provided. Try not to use 'printf' in generated code. Don’t make up APIs that don't exist.
        """

        text = f"{self.prompt} \n {self.code} \n The error report: \n {self.compile_result}\n The code fixes suggestions: \n {suggestions}\n{suggestions_2}"
        

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
        def robust_chat_completion(the_messages, the_model="gpt-4o"):
            response = openai.chat.completions.create(
                # model="gpt-4o-mini-2024-07-18",
                model=the_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": the_messages,
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
                                    "description": " the repaired code"
                                },
                            },
                            "required": [
                                "code",
                            ],
                            "additionalProperties": False
                        }
                    }
                },
                temperature=0.3,             ###  higher means more random
                max_tokens=2048,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )
            return response

        response = robust_chat_completion(the_messages=text, the_model="gpt-4o")

        content = response.choices[0].message.content

        try:
            data = json.loads(content)
            self.code = data["code"]
            save_harness(self.code_file, self.code)
        except Exception as e:
            print(e)


    def fix_warning(self, mode):

        self.prompt = """
    Read the code and the corresponding compilation warning message, then fix the code to reduce warning, making it compile correctly. 
    Try not to use 'printf' in generated code. Don't make up APIs that don't exist.
    
    For example, variable "xxx" is used before its value is set, the warning indicates that the variable has not been initialized correctly, 
    or that the parameter was passed using value copying, and a variable alias should be used instead &. E.g. UT(steam) should be modified by UT(&steam).
            """

        text = f"{self.prompt} \n {self.code} \n The error report: \n {self.compile_result}\n "

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
        def robust_chat_completion(the_messages, the_model="gpt-4o"):
            response = openai.chat.completions.create(
                # model="gpt-4o-mini-2024-07-18",
                model=the_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": the_messages,
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
                                    "description": " the repaired code"
                                },
                            },
                            "required": [
                                "code",
                            ],
                            "additionalProperties": False
                        }
                    }
                },
                temperature=0.3,  ###  higher means more random
                max_tokens=2048,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )
            return response
        response = robust_chat_completion(the_messages=text, the_model="gpt-4o")

        content = response.choices[0].message.content
        

        try:
            data = json.loads(content)
            self.code = data["code"]
        except Exception as e:
            print(e)

        save_harness(self.code_file, self.code)


# check the api sequences of generated code, return the api sequence list
def check_api_of_generated_code(code, api_list):
    truly_api_list = api_list[:]

    for api in api_list:
        if api not in code:
            truly_api_list.remove(api)

    return truly_api_list

class A_PATH:
    def __init__(self, **kwargs):
        self.code = None
        self.code_file = None
        self.compilation = None

        if 'prompt' in kwargs:
            self.prompt = kwargs['prompt']

        if 'analysis' in kwargs:
            self.analysis = kwargs['analysis']

def exec_multi_harness_para(exec_list,
                          round_list,
                          timelimit_per_program,
                          verbose_level,
                          fuzz_cpu_limit,
                          pre_fuzz_cpu_limit,
                          mode) -> list:

    current_directory = os.getcwd()
    if mode == 'rt':
        working_dir = f"{current_directory}/rt-lib/fuzz_output_threads"
    elif mode == 'nvjpeg':
        working_dir = f"{current_directory}/nvjpeg/fuzz_output_threads"
    elif mode == 'cublas':
        working_dir = f"{current_directory}/cublas/fuzz_output_threads"
    else:
        working_dir = f"{current_directory}/{mode}/fuzz_output_threads"

    return fuzz(
        target='cuda',  ##
        exec_list_path=exec_list,  ##
        round_list=round_list,
        timelimit_per_program=timelimit_per_program,  ##
        working_dir=working_dir,  ##
        verbose_level=verbose_level,  ##
        fuzz_cpu_limit=fuzz_cpu_limit,
        pre_fuzz_cpu_limit=pre_fuzz_cpu_limit
        # mode=mode
    )


def fuzz_scheduling(config : dict, bitmap_save_path, bitmap_load_path, max_fix, time_T, input_json : list, mode : str, mutator=False):
    """
    this function is to schedule other inner functions of this class, fulfilling the ability that:

    the call and order graph -> sample one path -> generate harness -> compile and debug -> extract call and order -> mutate harness -> compile and debug -> extract call and order
                ^                                                                                  |                                                                 |
                |                                                                                  |                                                                 |
                |                                                                          if new edge appears                                            if new edge appears
                |                                                                                  |                                                                 |
        update the graph  < - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    """
    total_har_count = 0                             ##  count the number of harness generated by llm and knowledge graph
    succ_har_count = 0                              ##  count the number of harness generated by llm and knowledge graph and compiled success
    succ_har_sep = 0                                ##  count the number of success harness by above step and separated success.
    succ_har_sep_wrap = 0                           ##  count the number of success harness by above step and wrapped success.
    all_harness_list = []                           ##  using a list, record all the harness generated and celibrate them

    LLM = Chatgpt()

    origin_G = establish_graph(input_json)
    ## delete self-loop edge
    self_loops = list(nx.selfloop_edges(origin_G))
    origin_G.remove_edges_from(self_loops)

    extend_cuda_graph = Extended_Graph(Origin_G = origin_G, load_map= bitmap_load_path, MAP_SIZE = 65536)
    extend_cuda_graph.display_graph()

    start_time = time.time()

    switch_init = 0
    mutate_switch = switch_init   # If two consecutive sets of samples are discarded, switch to mutation mode. that is mutate_switch > 9

    save_count = 1
    add_count = 0
    match = re.search(r'running_(\d+)_hours', bitmap_load_path)
    if match:
        add_count = int(match.group(1))
        print(f"Extracted number: {save_count}")
    else:
        print("No matching pattern found")

    while time.time() - start_time < time_T:
        print(time.time() - start_time)
        if time.time() - start_time > 3600 * save_count:
            ## save bitmap, every hour approximately.
            extend_cuda_graph.save_bitmap(bitmap_save_path, save_count+add_count)
            save_count += 1

        # if  len(all_harness_list) > 2 : ccc = 1
        ##  execute the harness mutation phase, rather than sample path from knowledge graph
        if mutator and mutate_switch > 9:
            mutate_switch = switch_init

            print(f'\n   @@@@@@@@@@@@@@@@ harness mutating phase @@@@@@@@@@@@@@@@@@@  \n')
            logging.info(f'\n   @@@@@@@@@@@@@@@@ harness mutating phase @@@@@@@@@@@@@@@@@@@  \n')
            index1 = get_one_harness_candi_mutate(all_harness_list)
            the_harness_seed = f'./harness/{all_harness_list[index1].ID}'
            # the_harness_seed = './harness/20250325_205433'

            count1 = 0
            succ_mutated = 0

            while(count1 < 3):    #  mutation up to 3 times
                mu_code, mu_compiler, mu_seq, directory_folder = mutate_harness(the_harness_seed)

                ## check for new edges occurrence
                the_button = False
                for i in range(len(mu_seq) - 1):
                    if mu_seq[i] not in extend_cuda_graph.api_key.keys():
                        extend_cuda_graph.add_new_api_key( mu_seq[i])

                    if mu_seq[i+1] not in extend_cuda_graph.api_key.keys():
                        extend_cuda_graph.add_new_api_key( mu_seq[i+1])

                    the_order_edge_key = (extend_cuda_graph.api_key[mu_seq[i]] >> 1) ^ \
                                     extend_cuda_graph.api_key[mu_seq[i + 1]]
                    if the_order_edge_key not in extend_cuda_graph.bitmap_api_order_edge_trace:
                        the_button = True  ##  that means a new order edge has been discovered, this path is valuable
                        # extend_cuda_graph.bitmap_api_order_edge_trace.add(the_order_edge_key)  # when compile successful ,add it

                if the_button:
                    mu_path = directory_folder
                    mu_name = f'{os.path.basename(the_harness_seed)}_mu.cu'
                    os.makedirs(mu_path, exist_ok=True)
                    shutil.copy(the_harness_seed+f'/{os.path.basename(the_harness_seed)}.cu', mu_path)
                    save_harness(os.path.join(mu_path, mu_name), mu_code)
                    mu_compiler2 = mu_compiler.replace(' a.cu', f' {os.path.join(mu_path, mu_name)}').replace(
                        ' a.out', f' {mu_path}/{mu_name[:-3]}').replace( 'nvcc', '/usr/local/cuda/bin/nvcc')

                    compile_command = shlex.split(mu_compiler2)
                    try:
                        result = subprocess.run(compile_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                                text=True,
                                                check=True)
                        print(f"   Mutated code Compilation successful: {compile_command}")
                        logging.info(f"   Mutated code Compilation successful: {compile_command}")
                        succ_mutated += 1
                        break

                    except subprocess.CalledProcessError as e:
                        print(f"   Mutated code Compilation failed: {compile_command}")
                        logging.info(f"   Mutated code Compilation failed: {compile_command}")
                        result = e.stderr
                count1 += 1

            if succ_mutated:     ##  means mutated code compilation successfully, and new edge appears.
                extend_cuda_graph.bitmap_api_order_edge_trace.add(the_order_edge_key)
                for i in range(len(mu_seq) - 1):
                    if mu_seq[i] not in extend_cuda_graph.origin_graph.nodes:
                        if i > 0:
                            extend_cuda_graph.update_origin_graph(mu_seq[i-1], mu_seq[i], '')
                        extend_cuda_graph.update_origin_graph('', mu_seq[i], mu_seq[i+1])

                if (harness_inti_separate(mu_code, os.path.join(mu_path, mu_name), mu_compiler2, mode, suffix='_sep')):
                    print(f'       [+]Next! this harness (+ separate) compile pass\n')
                    logging.info(f'       [+]Next! this harness (+ separate) compile pass\n')
                    new_name = f'{os.path.join(mu_path, mu_name)[:-3]}_sep.cu'

                    compile_command = shlex.split(mu_compiler2)
                    for i in range(len(compile_command)):
                        if "_" in compile_command[i]:
                            if ".cu" in compile_command[i]:
                                compile_command[i] = compile_command[i][:-3] + "_sep.cu"
                            else:
                                compile_command[i] = compile_command[i] + "_sep"

                    sep_compile_cmd = ' '.join(compile_command)
                    fixed_wrapper_code = parse_and_wrap_c_file(config, new_name)
                    save_harness(f'{new_name[:-3]}_wrap.cu', fixed_wrapper_code)

                    if compile_code(sep_compile_cmd, suffix='_wrap'):
                        succ_har_sep_wrap += 1
                        print(f'           [+]Next! this harness (+ separate + wrap) compile pass\n')
                        logging.info(f'           [+]Next! this harness (+ separate + wrap) compile pass\n')
                        generated_code_api_list = mu_seq
                        current_directory = os.getcwd()
                        exec_list_path = [mu_path]  ##  there are multi harness executed parallelly
                        round_list = list(range(1, 8))

                        if config['lib'] == 'nvjpeg':
                            timelimit_per_program = config['fuzz_one']['executing_time_first'] + 2
                        else:
                            timelimit_per_program = config['fuzz_one']['executing_time_first']
                        timelimit_per_program *= 30

                        verbose_level = 3
                        fuzz_cpu_limit = 20
                        pre_fuzz_cpu_limit = 5
                        print(f'    ******  test the harness current once ********** \n')
                        logging.info(f'    ******  test the harness current once ********** \n')
                        for i in range(1, 8, 1):
                            round_list = [i]
                            exec_result = exec_multi_harness_para(exec_list_path,
                                                                round_list,
                                                                timelimit_per_program,
                                                                verbose_level,
                                                                fuzz_cpu_limit,
                                                                pre_fuzz_cpu_limit,
                                                                mode)

                        ##  establish the "celibrate_harness" class, append the new class to all_harness_list
                        ttemp = celibrated_harness(the_time_flame=os.path.basename(mu_path),
                                                                   request_api_list=generated_code_api_list,
                                                                   generated_api_list=generated_code_api_list,
                                                                   exec_oracle=exec_result[0],
                                                                   round=7)
                        ttemp.mutate_add()
                        all_harness_list.append(ttemp)


                    else:
                        print(f'           [-]Next! this harness (+ separate + wrap) compile wrong\n')
                        logging.warning(f'           [-]Next! this harness (+ separate + wrap) compile wrong\n')
                else:
                    print(f'       [-]Next! this harness (+ separate) compile wrong\n')
                    logging.warning(f'       [-]Next! this harness (+ separate) compile wrong\n')



            print(f'\n   @@@@@@@@@@@@@@@@ harness mutating done once @@@@@@@@@@@@@@@@@@@  \n')
            logging.info(f'\n   @@@@@@@@@@@@@@@@ harness mutating done once @@@@@@@@@@@@@@@@@@@  \n')

        paths = get_paths_for_nodes_within_graph(extend_cuda_graph.origin_graph)  # this is a list with 5 elements where each one is a api sequence list

        if total_har_count % 10 == 0: mutate_switch = switch_init  #  Reset once every 10 sequences

        for one_api_sequence_list in paths:

            one_api_sequence_list = one_api_sequence_list[:5]     # just use first 5 api, too long, llms can not generate harness well!!!
            total_har_count += 1

            #one_api_sequence_list = ['cudaMemPoolGetAccess', 'cudaMemPoolSetAccess', 'cudaStreamWaitEvent', 'cudaMallocAsync', 'cudaMemPoolExportToShareableHandle', 'cudaMemPoolCreate']
            call_list, call_list_tuple = extend_cuda_graph.get_call_edge_attribute_for_one_path(one_api_sequence_list)    ##  gain the call relationship from a api path
            print(f'\n[*] the sample path is: {one_api_sequence_list}')
            logging.info(f'\n[*] the sample path is: {one_api_sequence_list}')

            ## check for new edges occurrence
            the_button = False
            for i in range(len(one_api_sequence_list) - 1):
                the_order_edge_key = (extend_cuda_graph.api_key[one_api_sequence_list[i]] >> 1) ^ extend_cuda_graph.api_key[one_api_sequence_list[i+1]]
                if the_order_edge_key not in extend_cuda_graph.bitmap_api_order_edge_trace:
                    the_button = True          ##  that means a new order edge has been discovered, this path is valuable
                    extend_cuda_graph.bitmap_api_order_edge_trace.add(the_order_edge_key)
                if call_list_tuple:
                    for one_call_tuple in call_list_tuple:
                        the_call_edge_key = (extend_cuda_graph.api_key[one_call_tuple[0]] >> 1) ^ extend_cuda_graph.api_key[one_call_tuple[1]]
                        if the_call_edge_key not in extend_cuda_graph.bitmap_api_call_edge_trace:
                            the_button = True  ##  that means a new call edge has been discovered, this path is valuable
                            extend_cuda_graph.bitmap_api_call_edge_trace.add(the_call_edge_key)


            if not the_button:
                mutate_switch += 1
                # for current list, check whether it has been executed before
                # and whether the execution results are consistent with expectations
                the_button2 = True

                cc = 0
                if all_harness_list:
                    for item in all_harness_list:
                        if item.request_api_list == one_api_sequence_list:
                            cc += 1
                            if not item.consistence:
                                the_button2 = False
                    if not cc:
                        the_button2 = False
                else:
                    the_button2 = False

                if the_button2:
                    print(f'   this path is valueless, abandon!\n')
                    logging.warning(f'   this path is valueless, abandon!\n')
                    continue

            max_retries = 5
            retry_count = 0
            code1, compilation = None, None

            while retry_count < max_retries:
                try:
                    code1, compilation = LLM.send_request(one_api_sequence_list, call_list, mode)
                    if code1 is not None and compilation is not None:  # 验证返回值有效性
                        break
                
                except (KeyError, AttributeError) as e:  # 捕获模型返回格式错误
                    print(f"模型返回异常: {e}, 第{retry_count + 1}次重试...")
                except Exception as e:  # 捕获其他未知错误
                    print(f"未知错误: {e}, 第{retry_count + 1}次重试...")

                retry_count += 1
                time.sleep(2 ** retry_count)  # 指数退避避免频繁

            if code1 is None:
                raise ValueError(f"超过最大重试次数{max_retries}，请求失败")

            # code1, compilation = LLM.send_request(one_api_sequence_list, call_list, mode)

            print(f'   the save file is: {LLM.code_file}')
            logging.info(f'   the save file is: {LLM.code_file}')
            print(f'   the compile is: {compilation}')
            logging.info(f'   the compile is: {compilation}')

            i = 0
            while ( i < max_fix and (not LLM.compile_code()) ):
                LLM.bugfix(mode)
                i += 1

            if LLM.compile_code():
                if LLM.compile_result:
                    if "warning" in LLM.compile_result.lower():
                        LLM.fix_warning(mode)

                call_list, _ = extend_cuda_graph.get_call_edge_attribute_for_one_path(one_api_sequence_list)
                #extend_cuda_graph.save_bitmap(path1)
                print(f'   :) Great! this harness (init) compile pass\n')
                logging.info(f'   :) Great! this harness (init) compile pass\n')
                succ_har_count += 1
                if (harness_inti_separate(LLM.code, LLM.code_file, LLM.compilation, mode, suffix='_sep')):
                    succ_har_sep += 1
                    print(f'       [+]Next! this harness (+ separate) compile pass\n')
                    logging.info(f'       [+]Next! this harness (+ separate) compile pass\n')
                    new_name = f'{LLM.code_file[:-3]}_sep.cu'

                    ## updata LLM.compilation
                    compile_command = shlex.split( LLM.compilation)
                    for i in range(len(compile_command)):
                        if "_" in compile_command[i]:
                            if ".cu" in compile_command[i]:
                                compile_command[i] = compile_command[i][:-3] + "_sep.cu"
                            else:
                                compile_command[i] = compile_command[i] + "_sep"
                    LLM.compilation = ' '.join(compile_command)

                    ## Unit Testing: Using a certain case to test the wrapper logic
                    # new_name = f'./harness/ttt/095227_sep.cu'
                    # LLM.compilation = '/usr/local/cuda/bin/nvcc ./harness/ttt/095227_sep.cu -o ./harness/ttt/095227_sep'
                    # LLM.folder_name = 'ttt'
                    ## Unit Testing: Using a certain case to test the wrapper logic

                    try:
                        # 尝试执行可能失败的函数
                        fixed_wrapper_code = parse_and_wrap_c_file(config, new_name)
                    except Exception as e:  # 捕获所有异常[1,3](@ref)
                        print(f"警告: parse_and_wrap_c_file 执行失败 ({e})，回退到直接读取文件")
                        try:
                            # 回退方案：直接读取文件内容
                            with open(new_name, 'r', encoding='utf-8') as f:  
                                fixed_wrapper_code = f.read()  
                        except Exception as fallback_error:
                            # 连文件读取也失败时的处理
                            print(f"严重错误: 回退方案失败 ({fallback_error})")
                            fixed_wrapper_code = ""  # 最终兜底值

                    save_harness(f'{new_name[:-3]}_wrap.cu', fixed_wrapper_code)
                    if compile_code(LLM.compilation, suffix='_wrap'):
                        succ_har_sep_wrap += 1
                        print(f'           [+]Next! this harness (+ separate + wrap) compile pass\n')
                        logging.info(f'           [+]Next! this harness (+ separate + wrap) compile pass\n')
                        generated_code_api_list = check_api_of_generated_code(LLM.code, one_api_sequence_list)

                        #  exec the harness bin parallelly, the source is from fuzz_exec_para.py/fuzz function.
                        current_directory = os.getcwd()
                        if mode == 'rt':
                            exec_list_path = [f"{current_directory}/rt-lib/harness/{LLM.folder_name}"]  ##[f"{current_directory}/harness/ttt", f"{current_directory}/harness/ttt1"]   ##  there are multi harness executed parallelly
                        elif mode == 'nvjpeg':
                            exec_list_path = [f"{current_directory}/nvjpeg/harness/{LLM.folder_name}"]
                        elif mode == 'cublas':
                            exec_list_path = [f"{current_directory}/cublas/harness/{LLM.folder_name}"]
                        else:
                            exec_list_path = [f"{current_directory}/{mode}/harness/{LLM.folder_name}"]

                        round_list = [1]    ## [1, 0]  # round times for one harness

                        if config['lib'] == 'nvjpeg':
                            timelimit_per_program = config['fuzz_one']['executing_time_first'] + 2
                        else:
                            timelimit_per_program = config['fuzz_one']['executing_time_first']
                        timelimit_per_program *= 60

                        verbose_level = 3
                        fuzz_cpu_limit = 20
                        pre_fuzz_cpu_limit = 5
                        print(f'    ******  test the harness current once ********** \n')
                        logging.info(f'    ******  test the harness current once ********** \n')
                        exec_result = exec_multi_harness_para(exec_list_path,
                                              round_list,
                                              timelimit_per_program,
                                              verbose_level,
                                              fuzz_cpu_limit,
                                              pre_fuzz_cpu_limit,
                                              mode)

                        ##  establish the "celibrate_harness" class, append the new class to all_harness_list
                        all_harness_list.append(celibrated_harness(the_time_flame=LLM.folder_name,
                                                                   request_api_list=one_api_sequence_list,
                                                                   generated_api_list=generated_code_api_list,
                                                                   exec_oracle=exec_result[0],
                                                                   round=round_list[0]))

                        ##   chose 5 harness to execute, according to celibration score and sampling algorithm
                        top_5_harness = get_top_indices(all_harness_list)
                        if mode == 'rt':
                            exec_list_path = [f"{current_directory}/rt-lib/harness/{all_harness_list[item].ID}" for item in top_5_harness]
                        else:
                            exec_list_path = [f"{current_directory}/{mode}/harness/{all_harness_list[item].ID}" for item in
                                              top_5_harness]
                        round_list = [all_harness_list[item].round + 1 for item in top_5_harness]
                        print(f'    ******  test the 5 harnesses sampled ********** \n')
                        logging.info(f'    ******  test the 5 harnesses sampled ********** \n')
                        exec_result = exec_multi_harness_para(exec_list_path,
                                                              round_list,
                                                              timelimit_per_program / 2,
                                                              verbose_level,
                                                              fuzz_cpu_limit,
                                                              pre_fuzz_cpu_limit,
                                                              mode)

                        for idx, item in enumerate(top_5_harness):
                            all_harness_list[item].update(exec_result[idx], new_round=1)



                    else:
                        print(f'           [-]Next! this harness (+ separate + wrap) compile wrong\n')
                        logging.warning(f'           [-]Next! this harness (+ separate + wrap) compile wrong\n')
                else:
                    print(f'       [-]Next! this harness (+ separate) compile wrong\n')
                    logging.warning(f'       [-]Next! this harness (+ separate) compile wrong\n')

            else:
                print(f'   :( this harness (init) compile wrong\n')
                logging.warning(f'   :( this harness (init) compile wrong\n')



    print(f'during {time_T / 60}min testing, {total_har_count} harness generated, \
and {succ_har_count} compiled successfully, the {succ_har_sep}data init success!')
    logging.info(f'during {time_T / 60}min testing, {total_har_count} harness generated, \
and {succ_har_count} compiled successfully, the {succ_har_sep}data init success!')
    print(all_harness_list)




if __name__ == '__main__':

    the_max_fix_num = config['harness']['the_max_fix_num']
    time_threshold = config['process']['time_threshold']

    mutator = config['harness']['mutator']

    #  means which library you are going to test, choose one from ['rt', 'nvjpeg', 'cublas', 'cunpp']
    mode = config['lib']
    temp = ''
    if mode not in ['rt', 'nvjpeg', 'cublas', 'cunpp', 'cufft', 'curand', 'cusolver', 'cusparse']:
        raise ValueError(" THE TESTED LIB HAS NOT BEEN INCLUDED YET! THE SUPPORTTED LIBS ARE ['rt', 'nvjpeg', 'cublas', 'cunpp', " \
        "'cufft', 'curand', 'cusolver', 'cusparse'] ")
    else:
        if mode == 'rt':
            input_json = config['input_json']['rt']
            temp = 'rt-lib'
        else:
            input_json = config['input_json'][mode]
            temp = mode

    the_continue_edge_file = config['process']['the_continue_edge_file']   ##f'./{temp}/bitmap/bitmap_running_1_hours.pkl'
    if the_continue_edge_file:
        the_continue_edge_file = f'./{temp}/bitmap/{the_continue_edge_file}'
    save_path_bitmap = f'./{temp}/bitmap/'

    if not os.path.exists(save_path_bitmap):
        # 如果文件夹不存在，则创建它
        os.makedirs(save_path_bitmap)
        print(f"folder {save_path_bitmap} has been created.")
    else:
        # 如果文件夹已经存在，跳过创建
        print(f"folder {save_path_bitmap} exists, skip creating.")

    fuzz_scheduling(config,
                    save_path_bitmap,
                    the_continue_edge_file,
                    the_max_fix_num,
                    time_threshold,
                    input_json,
                    mode,
                    mutator)



