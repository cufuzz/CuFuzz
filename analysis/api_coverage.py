import sys
import os
import re
## adding parents dir
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir) 

from establish_graph_from_json import establish_graph

from datetime import datetime


def slice_between(lines: list[str],
                  start_pattern: str,
                  end_pattern: str) -> list[str]:
    start_idx = end_idx = None
    for i, line in enumerate(lines):
        if line.lstrip().startswith(start_pattern) and start_idx is None:
            start_idx = i
        if line.lstrip().startswith(end_pattern):
            end_idx = i
            break
    if start_idx is None or end_idx is None or start_idx >= end_idx:
        return []

    # 只保留非纯注释行
    return [l for l in lines[start_idx + 1 : end_idx]
            if not l.lstrip().startswith('//')]



CU_FUNC_RE = re.compile(r'\b(cu\w+)\s*\(')
def extract_cu_funcs(segment: list[str]) -> list[str]:
    seen = set()
    result = []
    for line in segment:
        for match in CU_FUNC_RE.finditer(line):
            func = match.group(1)
            result.append(func)
            if func not in seen:
                seen.add(func)
    return result

def extract_ut_function_and_cu_calls(file_path):
    # 读取文件内容
    with open(file_path, 'r') as f:
        content = f.readlines()

    segment = slice_between(content, "void UT", "int main")
    cu_funcs = extract_cu_funcs(segment)
    return cu_funcs
    

def api_coverage_calcu(harness_root, minute_step):
    ###  read harness dir to be test.  traverse every haness(like  2025xxx_01xxx), read the code file,
    ###  gain api sequence leveraging re.
    ###  minute_step means how many minutes to record a point .

    result = {}
    time_list = []
    for harness_dir in os.listdir(harness_root):
        if "_" in harness_dir:
            one_harness_dir = os.path.join(harness_root, harness_dir)
            for item in os.listdir(one_harness_dir):
                if '_sep_wrap' in item and ('.cu' not in item):
                    ##  this is the traget file, harness wrapped successfully. read it and count the api sequence
                    timeflame = datetime.strptime(harness_dir, "%Y%m%d_%H%M%S")
                    result[timeflame] = ''
                    time_list.append(timeflame)

                    file_path = os.path.join(harness_root, harness_dir, harness_dir + '.cu')
                    # print(file_path)
                    
                    result[timeflame] = extract_ut_function_and_cu_calls(file_path)
                    # print(result[timeflame])


    time_list_sorted = sorted(time_list)

    api_count = [0]
    time_alx = [0]
    api_set = set()

    # print(time_list_sorted)
    point_count = (time_list_sorted[-1] - time_list_sorted[0]).total_seconds() / (minute_step *60)
    p_j = 1

    for i in range(len(time_list_sorted)):
        dd = (time_list_sorted[i] - time_list_sorted[0]).total_seconds()/60
        api_set.update(result[time_list_sorted[i]])
        print(f'{time_list_sorted[i]} || {result[time_list_sorted[i]]}')

        if dd > p_j * minute_step:
            time_alx.append(minute_step * p_j)
            api_count.append(len(api_set))
            p_j += 1

    time_alx.append( (time_list_sorted[-1] - time_list_sorted[0]).total_seconds() / float(60))
    api_count.append(len(api_set))

    print(time_alx)
    print(api_count)
    print(api_set)


                     




if __name__ == "__main__":
    json_path = ['./rt-lib/cudasample2json/gpt4o_output_calls_only_cu.json', \
                 './rt-lib/pdf2json/gpt4o_output_calls.json', \
                 './rt-lib/cudasample2json/gpt4o_output_order_only_cu.json', \
                 './rt-lib/pdf2json/gpt4o_output_order.json']

    #G = establish_graph(json_path)

    root_path = '../experiments/cu-cufft/cufft/harness'
    api_coverage_calcu(root_path, minute_step=25)

