##  this file for harness mutatation
import os
import shutil
import openai
from several_prompt import *
import json
import datetime
import yaml

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

PMT = PROMPTS()

def mutate_harness(the_name:str) -> str:
    ##  the_name likes './harness/20250325_205433', it is a folder path for a harness

    ## mkdir a new folder to save mutated harness
    now = datetime.datetime.now()
    timestemp = now.strftime("%H%M%S")
    directory_name = the_name + '_mutated'
    if os.path.exists(directory_name):
        # if exist, add a suffix
        directory_name += f'_{timestemp}'
    os.makedirs(directory_name, exist_ok=True)
    shutil.copy(the_name+f'/{os.path.basename(the_name)}.cu', directory_name)
    the_seed_path = f'{directory_name}/{os.path.basename(the_name)}.cu'

    with open(the_seed_path, 'r') as f:
        the_code = f.read()
    f.close()

    code_prompt = """
You are an expert agent specialized in cuda programming. Mutate the given CUDA code
If there are some bugs in the given code itself, fix them first and then mutate them.
        
The ways of mutation include but are not limited to the following:
Add some new CUDA APIs
Replace certain CUDA APIs
Change the type of the kernel function, such as __Device____ global__,  Conversion between waiting periods
        
Must note that the mutated code should be compiled successfully.
        
the codes need to be mutated are as follows:
        
%s
        """%(the_code)

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
    def robust_chat_completion(the_model="gpt-4o"):
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
                            "sequence": {
                                "type": "array",
                                "description": f"A list of APIs in the mutated codes.",
                                "items": {
                                    "type": "string"
                                }
                            },
                        },
                        "required": [
                            "code",
                            "compiler",
                            "sequence",
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
    response = robust_chat_completion(the_model="gpt-4o")

    content = response.choices[0].message.content
    try:
        data = json.loads(content)
    except Exception as e:
        print(f"error:{e}")
        print(content)

    new_code = data["code"]
    new_compilation = data["compiler"]
    new_seq = data["sequence"]



    return new_code, new_compilation, new_seq, directory_name




if __name__ == '__main__':
    the_harness_seed = './harness/20250325_205433'
    mutate_harness(the_harness_seed)