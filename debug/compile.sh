#!/bin/bash

# 用法提示
if [ $# -lt 1 ]; then
    echo "Usage: $0 <timestamp_id> [-l<lib>] [-l <lib>] ..."
    echo "Example: $0 ../cublas/harness/20250615_111700 -lcublas -lcurand -l nvjpeg"
    exit 1
fi

# 取 target_dir & timestamp_id
target_dir=$1
timestamp_id=$(basename "$target_dir")
shift  # 剩余参数用于解析 -l*

# 推荐：用数组保存全部库
library_flags=()

# 解析所有剩余参数
while [ $# -gt 0 ]; do
  case "$1" in
    -l)  # 处理分隔写法：-l foo
      if [ $# -ge 2 ] && [[ ! "$2" =~ ^- ]]; then
        library_flags+=("-l$2")
        shift
      else
        echo "Warning: '-l' provided without a library name; ignored"
      fi
      ;;
    -l*) # 处理连写：-lfoo
      library_flags+=("$1")
      ;;
    *)
      # 如需支持其他参数（-L/-I 等），可在此扩展；当前忽略非 -l 参数
      echo "Warning: ignoring non -l argument '$1'"
      ;;
  esac
  shift
done

# 兼容你原先的字符串变量（如果后面仍按字符串拼接）
if ((${#library_flags[@]})); then
  library_flag="${library_flags[*]}"
  echo "Linking libraries: ${library_flags[*]}"
else
  library_flag=""
  echo "No compiling libraries specified"
fi

# 构建编译命令
compile_command="nvcc -g -G ${target_dir}/${timestamp_id}_sep_wrap.cu ../c_factors/mu2.o -o ${target_dir}/db_wrap"

AFL_PATH=$(python3 - <<'PY'
import re, sys
with open('../config.yaml','r',encoding='utf-8') as f:
    s = f.read()
m = re.search(r'(?m)^\s*AFL_PATH\s*:\s*(.+?)\s*$', s)
print(m.group(1).strip(" '\"") if m else "")
PY
)

# 兜底校验
[ -n "$AFL_PATH" ] || { echo "ERROR: AFL_PATH not found in ../config.yaml"; exit 1; }

# 组装最终编译命令：添加 -I"$AFL_PATH"
compile_command="nvcc -g -G -I\"$AFL_PATH\" ${target_dir}/${timestamp_id}_sep_wrap.cu ../c_factors/mu2.o -o \"${target_dir}/db_wrap\""



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
