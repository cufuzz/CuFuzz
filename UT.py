###  this file for establishing some unit testing function.
###  for instance, test harness_inti_separate()

from fuzz_one import *
import re

def celibate_harness_init():
    ## UT for harness_inti_separate()
    time_flame = '20250626_140703'
    in_file = f'cublas/harness/{time_flame}/{time_flame}.cu'
    with open(in_file, 'r') as f:
        content = f.read()
    f.close()

    mu_compiler2 = f'/usr/local/cuda/bin/nvcc cublas/harness/{time_flame}/{time_flame}.cu -o cublas/harness/{time_flame}/{time_flame} -lcublas'
    button = harness_inti_separate(content, in_file, mu_compiler2, 'cublas', suffix='_sep')
    print(button)

def check_timeout(new_one):
    ## check whether the timeout case has been tested, maintain a set()
    the_pool = {'123303', '121849', '121800', '123520', '122920', '123824', '122838', '122605', '124202', '124303', '124602', '120523'}
    print( new_one in the_pool)

def search_cuda_t(file_path):
    """
    提取.cu文件中的xxx_t类型变量及其相关语句
    :param file_path: .cu文件路径
    :return: 字典{变量名: [包含该变量的语句列表]}
    """
    # 初始化结果字典
    result_dict = {}
    
    # 正则表达式匹配xxx_t类型变量声明
    # 匹配模式：可选修饰符（如static） + 类型名（xxx_t） + 变量名（可能带指针*）
    pattern = r'(?:\b\w+\s+)*(\b\w+_t)\s+(\*?\s*\w+)\s*(?:=|\[|;|,)'

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.readlines()
            
            # 第一次遍历：识别所有xxx_t类型变量
            for line in content:
                matches = re.finditer(pattern, line)
                for match in matches:
                    var_type = match.group(1)  # 变量类型（如nvjpegDevAllocator_t）
                    var_name = match.group(2).strip()  # 变量名（如dev_allocator）
                    # 处理指针情况（如 *p 转为 p）
                    var_name = var_name.replace('*', '').strip()
                    if var_name not in result_dict:
                        result_dict[var_name] = []
            
            # 第二次遍历：收集每个变量的相关语句
            for line in content:
                for var_name in result_dict.keys():
                    # 如果行包含变量名且不是纯注释（简单过滤）
                    if var_name in line and not line.strip().startswith('//'):
                        result_dict[var_name].append(line.strip())
    
    except FileNotFoundError:
        print(f"错误：文件 {file_path} 不存在")
        return {}
    except Exception as e:
        print(f"发生错误：{str(e)}")
        return {}
    
    for var_name, statements in result_dict.items():
        print(f"变量名: {var_name}")
        print("相关语句:")
        for stmt in statements:
            print(f"  - {stmt}")
        print("-" * 50)

    return result_dict

if __name__ == "__main__":
    celibate_harness_init()
    #check_timeout('120523')

    # search_cuda_t(f'../CUDALibrarySamples/nvJPEG/Image-Resize/imageResize.cpp')