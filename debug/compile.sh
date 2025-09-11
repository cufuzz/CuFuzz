#!/bin/bash

# 检查是否提供了至少一个参数
if [ $# -lt 1 ]; then
    echo "Usage: $0 <timestamp_id> [-l<library>]"
    echo "Example: $0 ./compile.sh ../cublas/harness/20250615_111700 -lcublas"
    exit 1
fi

# 获取传入的时间戳 ID
target_dir=$1
timestamp_id=$(basename "$target_dir")

# 初始化编译库标志
library_flag=""

# 检查是否有第二个参数（库链接选项）
if [ $# -ge 2 ]; then
    # 检查第二个参数是否以 -l 开头
    if [[ $2 == -l* ]]; then
        library_flag=$2
        echo "Linking library: $library_flag"
    else
        echo "Warning: Second argument should be a library link option (e.g. -lnvjpeg)"
    fi
else
    echo "No compiling library specified"
fi

# 构建编译命令
compile_command="nvcc -g -G ${target_dir}/${timestamp_id}_sep_wrap.cu ../c_factors/mu2.o -o ${target_dir}/db_wrap"

# 如果有库链接选项，则添加到编译命令
if [ -n "$library_flag" ]; then
    compile_command="$compile_command $library_flag"
fi

# 执行编译命令
echo "Executing: $compile_command"
eval $compile_command

# 检查编译是否成功
if [ $? -eq 0 ]; then
    echo "Compilation successful."
else
    echo "Compilation failed."
    exit 1
fi
