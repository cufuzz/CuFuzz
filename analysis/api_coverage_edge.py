from api_coverage import *

def extract_edges_from_sequence(api_sequence):
    """
    从一个API调用序列中提取所有的有向边。

    参数:
    api_sequence: list, 包含API名称的列表，顺序代表调用顺序。

    返回:
    set: 一个包含所有边的集合，每个边用元组 (source, target) 表示。
    """
    edges = set()
    for i in range(len(api_sequence) - 1):
        source = api_sequence[i]
        target = api_sequence[i + 1]
        edge = (source, target)
        edges.add(edge) # 使用集合自动去重本列表内的重复边
    return edges


def count_unique_edges(list_of_sequences):
    """
    统计多个API序列列表中所有不重复的有向边。

    参数:
    list_of_sequences: list of lists, 多个API序列列表组成的列表。

    返回:
    tuple: (所有不重复边的集合, 不重复边的总数)
    """
    all_unique_edges = set()
    for seq in list_of_sequences:
        edges_in_this_seq = extract_edges_from_sequence(seq)
        all_unique_edges |= edges_in_this_seq # 使用集合的并集操作合并边

    total_unique_edges = len(all_unique_edges)
    return all_unique_edges, total_unique_edges


def api_edge_coverage_calcu(harness_root, minute_step=60):
    result = {}
    time_list = []

    all_sequences = []
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
    api_edge_set = set()

    # print(time_list_sorted)
    point_count = (time_list_sorted[-1] - time_list_sorted[0]).total_seconds() / (minute_step *60)
    p_j = 1

    for i in range(len(time_list_sorted)):
        dd = (time_list_sorted[i] - time_list_sorted[0]).total_seconds()/60
        api_edge_set |= (extract_edges_from_sequence(result[time_list_sorted[i]]))
        #print(f'{time_list_sorted[i]} || {result[time_list_sorted[i]]}')

        if dd > p_j * minute_step:
            time_alx.append(minute_step * p_j)
            api_count.append(len(api_edge_set))
            p_j += 1

    time_alx.append( (time_list_sorted[-1] - time_list_sorted[0]).total_seconds() / float(60))
    api_count.append(len(api_edge_set))
    
    print(time_alx)
    print(api_count)
    




if __name__ == "__main__":
    json_path = ['./rt-lib/cudasample2json/gpt4o_output_calls_only_cu.json', \
                 './rt-lib/pdf2json/gpt4o_output_calls.json', \
                 './rt-lib/cudasample2json/gpt4o_output_order_only_cu.json', \
                 './rt-lib/pdf2json/gpt4o_output_order.json']

    #G = establish_graph(json_path)

    root_path = '../experiments/cu-cusparse/cusparse/harness'
    api_edge_coverage_calcu(root_path, minute_step=50)