import subprocess
import os
import json
import networkx as nx
import matplotlib.pyplot as plt
import random
from several_prompt import *
import openai
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

def check_api_in_cuda_a(api_name, the_set):
    """
    查看当前api_name是否存在与 cuda 目录中的.a 的函数签名集合中
    """
    if api_name in the_set:
        # print(f"API {api_name} found in the set.")
        return True
    else:
        # print(f"API {api_name} not found in the set.")
        return False


def get_all_signature_from_cuda_a(cuda_a_folder = '/usr/local/cuda/lib64') -> set:
    """
    用 nm 检索 cuda的所有 .a .so 文件中 api_name，并存为一个无重复的列表或其他数据格式
    """
    # 检查提供的文件夹路径是否有效
    if not os.path.isdir(cuda_a_folder):
        raise FileNotFoundError(f"Error: {cuda_a_folder} is not a valid directory.")  # 抛出错误并中断程序

    # 存储所有唯一 API 名称的集合
    api_signatures = set()

    # 遍历文件夹，查找所有的 .a 文件
    archive_files = []
    for root, _, files in os.walk(cuda_a_folder):
        for file in files:
            if file.endswith('.a') or file.endswith('.so'):  # 只处理 .a, .so 文件
                archive_files.append(os.path.join(root, file))

    # 遍历所有 .a 文件，使用 nm 检索 API
    for archive_file in archive_files:
        try:
            # 使用 subprocess 调用 nm 命令，传入 .a 文件路径
            result = subprocess.run(['nm', '-C', archive_file], capture_output=True, text=True, check=True)

            if archive_file .endswith('.so'):
                result = subprocess.run(['nm', '-C', '-D', archive_file ], capture_output=True, text=True, check=True)
            else:
                result = subprocess.run(['nm', '-C', archive_file ], capture_output=True, text=True, check=True)

            # 从 nm 输出中提取 API 名称
            for line in result.stdout.splitlines():
                parts = line.split()
                if len(parts) >= 3:  # 确保行包含符号信息
                    api_name = parts[2]  # 第三列是符号名称
                    api_signatures.add(api_name)  # 将 API 名称添加到集合中
        except subprocess.CalledProcessError as e:
            print(f"Error executing nm on {archive_file}: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    # 返回包含所有唯一 API 名称的集合
    return api_signatures


def establish_graph(json_files:list):
    """
    对于gpt4o 提取的call graph, order graph, 对其中的节点进行查看，用cuda官方的.h进行审查，
    只保留能在.h中查找到的api作为图的节点
    """

    G = nx.DiGraph()
    plt.figure(figsize=(20, 20), dpi=300)  # 图像大小为20x20英寸，分辨率为300 DPI
    api_signature_name_set = get_all_signature_from_cuda_a()

    for json_one in json_files:
        with open(json_one, 'r', encoding='utf-8') as file:

            for count, line in enumerate(file):

                # if count > 20: break
                try:
                # 尝试解析每一行为JSON对象
                    data = json.loads(line)
                    # print(data)
                    # 打印读取到的数据
                    # meta_path = data['cu_path']
                    llm_response = data['model_response']    ##  it is a list
                    if 'calls' in json_one:
                        for i in range(len(llm_response)):
                            head = llm_response[i]['head']
                            head_type = llm_response[i]['head_type']
                            description = llm_response[i]['description']
                            relation = llm_response[i]['relation']
                            tail = llm_response[i]['tail']
                            tail_type = llm_response[i]['tail_type']
                            if check_api_in_cuda_a(head, api_signature_name_set):
                                G.add_node(head)
                            if check_api_in_cuda_a(tail, api_signature_name_set):
                                G.add_node(tail)
                            if relation and check_api_in_cuda_a(head, api_signature_name_set) and \
                                    check_api_in_cuda_a(tail, api_signature_name_set):
                                G.add_edge(head, tail, attr1='call')

                    if 'order' in json_one:
                        order_list = llm_response['order']
                        for i in range(len(order_list)):
                            for j in range(len(order_list[i])):
                                if check_api_in_cuda_a(order_list[i][j], api_signature_name_set):
                                    G.add_node(order_list[i][j])

                                if j+1 < len(order_list[i]):
                                    if check_api_in_cuda_a(order_list[i][j + 1], api_signature_name_set):
                                        G.add_node(order_list[i][j + 1])
                                    if check_api_in_cuda_a(order_list[i][j], api_signature_name_set) and \
                                            check_api_in_cuda_a(order_list[i][j+1], api_signature_name_set):
                                        G.add_edge(order_list[i][j], order_list[i][j+1], attr2='order')



                except json.JSONDecodeError as e:
                    # 如果解析出错，打印错误信息
                    print(f"Error parsing JSON line: {e}")

        print(f"\nTotal number of valid JSON lines: {count + 1}, json name: {json_one}")
        first_edge_count = sum(1 for u, v, data in G.edges(data=True) if data.get('attr1') == 'call')
        second_edge_count = sum(1 for u, v, data in G.edges(data=True) if data.get('attr2') == 'order')

        print("节点:", len(G.nodes))
        print(f"边总数:, {len(G.edges)} || call 边数量：{first_edge_count} || order 边数量：{second_edge_count}")

        print("\n所有节点名称:")
        nodes_per_line = 10
        for i, node in enumerate(G.nodes, 1):
            print(node, end=", " if i % nodes_per_line != 0 else "\n")  # 每 10 个换行
        if len(G.nodes) % nodes_per_line != 0:  # 最后一行补换行（避免后续输出粘连）
            print()

    nx.draw(G, with_labels=True, node_size=100, node_color='lightblue', arrows=True, font_size=4)

    # plt.savefig("large_graph.png", format="png", bbox_inches="tight")  # 保存为PNG图像
    # plt.show()
    print('all done!')
    return  G

def calculate_node_and_path(Graph):
    """
    从获得的networkx图中搜索 孤立节点 、路径、 循环结构
    """
    nodes = list(Graph.nodes)
    isolated_nodes = [node for node in nodes if Graph.degree(node) == 0]
    print(f"孤立节点: {len(isolated_nodes)}个, 前十个是： {isolated_nodes[:10]}")
    # for u, v, data in Graph.edges(data=True):
    #     print(f"Edge ({u}, {v}) attributes: {data}")

def iterator_sample(successors:dict, node:str, path_list:list) -> list:
    if not node :
        return path_list
    else:
        if node in list(successors.keys()):
            if successors[node]:
                new_node = random.choice(successors[node])
                path_list.append(new_node)
            else:
                new_node = None
        else:
            new_node = None
        return iterator_sample(successors, new_node, path_list)


def get_paths_frome_networkx_success_dict(source_node:str, successors:dict, sample_numble:int = 5) -> list:

    all_sample_paths = []
    for i in range(sample_numble):
        one_sample_path = []
        one_sample_path.append(source_node)
        ##  sampling with a recursive function is essentially a depth-first search
        one_sample_path = iterator_sample(successors, source_node, one_sample_path)

        print(one_sample_path)
        all_sample_paths.append(one_sample_path)
    return all_sample_paths


def get_paths_for_nodes_within_graph(G):
    """ sample a node randomly, then sample 5 paths beginning from the node"""
    nodes = list(G.nodes)
    isolated_nodes = [node for node in nodes if G.degree(node) == 0]

    all_search_paths = []

    while True:
        node = random.choice(nodes)
        out_degree = G.out_degree(node)
        in_degree = G.in_degree(node)

        if out_degree > 1:
            successors = dict(nx.bfs_successors(G, node))   ##  the first key of dict is the source_node
            # print(successors)

            all_search_paths = get_paths_frome_networkx_success_dict(node, successors)

            break

    return all_search_paths


def gen_harness_for_api_senquence(paths, sys_prompt):
    for path in paths[:2]:
        openai.base_url = "https://api.gpt.ge/v1/"
        openai.default_headers = {"x-foo": "true"}

        user_prompt = "the cuda api suquence is: "
        for cuda_api in path:
            user_prompt = user_prompt + cuda_api + ', '


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
            completion = openai.chat.completions.create(
                model=the_model,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return completion

        response_content = robust_chat_completion(the_model="gpt-4o")
        
        print(response_content)



if __name__ == "__main__":
    import pickle

    with open('./rt-lib/pdf2json/rt_api_manual_dict.pkl', 'rb') as f:
        rt_api_signature_dict = pickle.load(f)

    for k,v in rt_api_signature_dict.items():
        print(f'{k} : {v}')
    if 'nvjpegEncodeGetBufferSize' in rt_api_signature_dict:
        print(1111111)

    # api_set = get_all_signature_from_cuda_a()
    # print(api_set)
    # print('\n\n')
    # for aaa in api_set:
    #     if 'nvjpegEncodeGetBufferSize' in aaa : print(aaa)


    print(1)


    json_path = ['./rt-lib/cudasample2json/gpt4omini_output_calls_only_cu.json', \
                 './rt-lib/pdf2json/gpt4omini_output_calls.json', \
                 './rt-lib/cudasample2json/gpt4omini_output_order_only_cu.json', \
                 './rt-lib/pdf2json/gpt4omini_output_order.json']

    G = establish_graph(json_path)
    # calculate_node_and_path(G)
    paths = get_paths_for_nodes_within_graph(G)

    prompts = PROMPTS()
    system_prompt = prompts.gen_harness_with_a_api_senquence
    gen_harness_for_api_senquence(paths, system_prompt)
