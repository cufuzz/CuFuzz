#!/bin/sh
# build.sh : compiling mutator.c → mu2.o then test it.
# usage: ./build.sh <AFL_INCLUDE_DIR>

AFL_PATH="$1"          # 第一个参数即为 AFL 头文件目录 the first para is the AFL absolute path

# 1.  check parameter
if [ -z "$AFL_PATH" ]; then
    echo "usage: $0 <AFL_INCLUDE_DIR>"
    echo "example: $0 /home/fanximing"
    exit 1
fi

if [ ! -d "$AFL_PATH" ]; then
    echo "err: dir $AFL_PATH is not exist"
    exit 2
fi

# 2. compiling target file
echo "Compiling mutator.c → mu2.o ..."
gcc -I"$AFL_PATH" -c mutator.c -o mu2.o
if [ $? -ne 0 ]; then
    echo "Failed to compile mutator.c"
    exit 3
fi

# 3. linking testing program
echo "Linking test executable 111 ..."
gcc -I"$AFL_PATH" main.c mu2.o -o 111
if [ $? -ne 0 ]; then
    echo "Failed to link test executable"
    exit 4
fi

echo "All done!"