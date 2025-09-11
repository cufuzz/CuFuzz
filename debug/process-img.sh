#!/bin/bash

##  given the harness path under test, process the orgin binary and _sep_wrap using compute-sanitizer and testing directly.
##  that is 4 results will be displayed. the difference between testing directly and extrnal sanitizer could be checked. 
# 检查是否提供了文件夹参数
if [ $# -eq 0 ]; then
    echo "错误：请提供目标文件夹路径作为参数"
    exit 1
fi

target_dir="$1"

# 检查文件夹是否存在
if [ ! -d "$target_dir" ]; then
    echo "错误：文件夹 '$target_dir' 不存在"
    exit 1
fi

# 从路径中提取文件夹名（最后一个/后的内容）
dir_name=$(basename "$target_dir")

# 定义两个可执行文件名
executable1="$dir_name"
executable2="${dir_name}_sep_wrap"

# 函数：执行命令并显示最后30行
run_and_tail() {
    local cmd="$1"
    echo "执行命令: $cmd"
    echo "----------------------------------------"
    $cmd | tail -n 30
    echo "----------------------------------------"
    echo ""
}

# 执行第一个可执行文件
if [ -f "$target_dir/$executable1" ]; then
    run_and_tail "$target_dir/$executable1 ../nvjpeg/harness/OIP.jpg"
else
    echo "警告：可执行文件 '$executable1' 不存在"
fi

# 使用compute-sanitizer执行第一个可执行文件
if [ -f "$target_dir/$executable1" ]; then
    run_and_tail "compute-sanitizer $target_dir/$executable1 ../nvjpeg/harness/OIP.jpg"
else
    echo "警告：无法用compute-sanitizer执行 '$executable1'，文件不存在"
fi

# 执行第二个可执行文件
if [ -f "$target_dir/$executable2" ]; then
    run_and_tail "$target_dir/$executable2 ../nvjpeg/harness/OIP.jpg"
else
    echo "警告：可执行文件 '$executable2' 不存在"
fi



# 使用compute-sanitizer执行第二个可执行文件
if [ -f "$target_dir/$executable2" ]; then
    run_and_tail "compute-sanitizer $target_dir/$executable2 ../nvjpeg/harness/OIP.jpg"
else
    echo "警告：无法用compute-sanitizer执行 '$executable2'，文件不存在"
fi

echo "所有操作完成"
