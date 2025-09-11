// main.c

#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include "mutate.h"


int main() {
    
    int a1 = 6;
    u32 a1_len = sizeof(a1);
    
    u8 *a1_buf = malloc(a1_len);
    memcpy(a1_buf, &a1, a1_len);
    
    int *h_data = malloc(a1 * 4);
    for (int i = 0; i < a1; i++) {
        h_data[i] = i;
    }

    u32 a2_len = a1 * 4;
    u8 *a2_buf = (u8 *)h_data;

    u32 new_len2 = random_havoc(a2_buf, a2_len, 1);

    u32 new_len = random_havoc(a1_buf, a1_len, 0); // 变异字节流

    a1 = *(int *)a1_buf;

    // // 释放内存
    free(h_data);

    return 0;
}


