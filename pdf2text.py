import PyPDF2
import re
import pickle

def pdf_to_text_one_page(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        meta = reader.metadata
        # print(len(reader.pages))
        for page in reader.pages:
            text += reader.pages[0].extract_text()
    return text

def pdf_to_text_whole(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ' '
        meta = reader.metadata
        # print(len(reader.pages))
        for page in reader.pages:
            text += page.extract_text()
    return text

def pdf_to_text_pages(pdf_path, page_start: int, page_end: int):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ' '
        meta = reader.metadata
        # print(len(reader.pages))
        for page in range(len(reader.pages)):
            if page in range(page_start, page_end, 1):
                text += reader.pages[page].extract_text() + '\n'
    return text

def read_cu_file(file_path):
    try:
        with open(file_path, 'r') as file:
            content = file.read()
        return content
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def runtime_api_manual2text(path='./cuda-pdf/CUDA_Runtime_API.pdf'):
    """
    read the cuda manual extract the api structure as a txt
    cuda runtime api manual: 10-490 pages  (others are
    """
    api_signature_list = []
    catalog_txt = pdf_to_text_pages(path, 1, 14)

    pattern = r"cuda\w+.*?(\d+)"
    function_line_tuples = []
    for line in catalog_txt.split('\n'):
        match = re.search(pattern, line)
        if match:
            # 提取函数名和行号
            function_name = match.group(0).split(maxsplit=1)[0]  # 分割匹配结果，取第一部分（函数名）
            line_number = match.group(1)  # 行号
            function_line_tuples.append((function_name, int(line_number)))

    # for i, (func, line) in enumerate(function_line_tuples):
    #     print(f"({i}:, {func}, {line})")                      ## the first 429 is valid

    for i, (func, line) in enumerate(function_line_tuples[:492]):
        page_txt = pdf_to_text_pages(path, line+31, line+32)

        pattern = r"cuda\w+\s*\(([^)]*?)\)"
        for match in re.finditer(pattern, page_txt, re.DOTALL):
            # 获取函数名
            function_name = match.group(0).split('(')[0].strip()
            # 获取括号内的内容，可能包含换行符
            params = match.group(1).strip()
            # 构造完整的API函数签名
            signature = f"{function_name}({params})".replace("\n", " ")
            api_signature_list.append(signature)

    # for signature in api_signature_list:
    #     print(signature)

    with open('./pdf2json/rt_api_manual.pkl', 'wb') as f:
        pickle.dump(api_signature_list, f)

    return api_signature_list

def rt_api_manual_list2dict():
    """
    此函数将cuda runtime api获得的签名列表转化为方便检索的字典，删除了重复项，一些 api(C API) api(C++ API)这种无意义项
    这个函数在发布版中并不直接使用
    """
    with open('./pdf2json/rt_api_manual.pkl', 'rb') as f:
        loaded_list = pickle.load(f)

    rt_api_dict = {}
    pattern = r"(\w+)(?=\()"
    for i , func in enumerate(loaded_list):
        # print(f'{i} {func}')
        match = re.search(pattern, func)
        if match:
            # 提取函数名
            function_name = match.group(1)
            rt_api_dict[function_name] = []

    for i, func in enumerate(loaded_list):
        match = re.search(pattern, func)
        if match:
            # 提取函数名
            function_name = match.group(1)
            rt_api_dict[function_name].append(func)


    for i, (k, v) in enumerate(dict(rt_api_dict).items()):
        # print(f'{i} {k}: {v}')
        for item in v[:]:
            match = re.search(r'\(([^)]*)\)', item)
            if match:
                # 括号内的内容
                content_inside_parentheses = match.group(1).strip()

                # 如果括号内没有字符或只有空格，则删除这个元素
                if not content_inside_parentheses:
                    rt_api_dict[k].remove(item)

    for i, (k, v) in enumerate(dict(rt_api_dict).items()):
        rt_api_dict[k] = list(set(v))

    api_substrings = ('(C API)', '(C++ API)')
    for i, (k, v) in enumerate(dict(rt_api_dict).items()):
        for item in v[:]:
            if any(substring in item for substring in api_substrings):
                rt_api_dict[k].remove(item)

    for i, (k, v) in enumerate(dict(rt_api_dict).items()):
        if len(v) > 1:
            print(f'{i} {k}: {v}')

    with open('./pdf2json/rt_api_manual_dict.pkl', 'wb') as f:
        pickle.dump(rt_api_dict, f)





if __name__ == "__main__":

    with open('./pdf2json/rt_api_manual_dict.pkl', 'rb') as f:
        loaded_dict = pickle.load(f)

    for i, (k, v) in enumerate(dict(loaded_dict).items()):

        print(f'{i} {k}: {v}')