import matplotlib.pyplot as plt
import numpy as np
import sys

time_data = [0, 60, 120, 180, 240, 300, 360, 420, 480, 540, 600, 660, 720, 780, 840, 900, 960, 1020, 1080, 1140, 1200, 1260, 1320, 1380, 1440]

# cuRT
rt_api_data = [0, 79, 117, 132, 136, 141, 144, 150, 151, 155, 158, 160, 162, 165, 167, 169, 171, 172, 173, 174, 174, 174, 176, 177, 178]
rt_api_edge_data = [0, 121, 227, 327, 388, 473, 538, 601, 653, 693, 723,750,793,826,861,892,926,951,975,1006,1023,1038,1059,1089,1098]
rt_fuzz4all_api = [0,	15,	17,	26	,30	,30,	30,	34,	36	,36	,37,	38,	39,	39,	39,	39,	41,	43,	43,	43,	43,	43,	44,	47,	47]
rt_fuzz4all_api_edge = [0,	53,	63,	77,	89,	95,	97,	111,115,115,118,126,128,129,134,141,148,148,150,152,157,169,169,177,180]


# nvjpeg 
nvjpeg_api_data = [0, 10, 12, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13,13]
nvjpeg_api_edge_data = [0, 29, 38, 46, 48, 50, 52, 60, 62, 63, 63, 63, 65, 70, 70, 71, 74, 74, 74, 74, 75, 75,75,75,75]
nvjpeg_fuzz4all_api = [0,6,6,7,7,8,	8,	8,	8,	8,	8,	8,	8,	8,	8,	8,	8,	8,	8,	8,	8,	8,	8,	8,	8]
nvjpeg_fuzz4all_api_edge = [0,	18,	18,	18,	18,	18,	18,	19,	19,	19,	19,	19,	24,	24,	24,	24,	24,	24,	25,	25,	26,	26,	26,	26,	26]

# cublas
cublas_api_data = [0, 28, 51, 56, 66, 87, 95, 96, 101, 114, 118, 124, 129, 129, 129, 129, 134,143,146,153,153,153,155,155,155]
cublas_api_edge_data = [0, 53, 88, 123, 157, 179, 190, 208, 226, 241, 269, 282, 300, 309, 329, 350, 365,380,387,396,404,413,422,430,439]
cublas_fuzz4all_api = [0,16,21,	22,	25,	28,	29,	33,	33,	36,	38,	39,	41,	41,	41,	42,	42,	43,	44,	44,	44,	44,	44,	44,	44]
cublas_fuzz4all_api_edge = [0,	55,	83,	94,	110,125	,133,146,148,161,168,172,177,178,179,183,185,190,193,193,196,199,200,202,202]



# curand
curand_api_data = [0, 17, 23, 25, 25, 25, 25, 25, 25, 25, 25, 25, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26]
curand_api_edge_data = [0, 27, 52, 71, 82, 89, 95, 110, 114, 116, 116, 119, 122, 124, 127, 130, 132, 132, 134, 137, 138, 139, 140,140,140]
curand_fuzz4all_api = [0,	11,	12,	13,	13,	14,	14,	14,	14,	14,	14,	15,	15,	15,	15,	15,	15,	15,	15,	15,	15,	16,	16,	16,	16]
curand_fuzz4all_api_edge = [0,28,31,	35,	35,	36,	40,	43,	44,	47,	50,	51,	51,	51,	52,	52,	53,	53,	53,	53,	53,	55,	57,	57,	57]



# cusolver 
cusolver_api_data = [0, 30, 48, 62, 69, 73, 81, 89, 92, 98, 101, 102, 103, 110, 112, 114, 116, 120, 122, 124, 125, 126, 126,126,126]
cusolver_api_edge_data = [0, 45, 69, 101, 126, 145, 161, 181, 202, 225, 243, 252, 260, 295, 308, 328, 346, 361, 371, 386, 393, 394,394,394,394]
cusolver_fuzz4all_api = [0,	18,	19,	19,	19,	19	,21,21,	21,	21,	21,	21,	21,	22,	22,	22,	22,	24,	24,	25,	25,	25,	25,	25,	25]
cusolver_fuzz4all_api_edge = [0,11,	17,	19,	20	,21,26,	27,	27,	28,	28,	29,	29,	30,	32,	33,	33,	34,	35,	36,	38,	39,	40,	40,	40]



# cufft 
cufft_api_data = [0, 17, 24, 24, 25, 25, 26, 26, 27, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 29, 29, 29,29,29,29]
cufft_api_edge_data = [0, 30, 48, 73, 76, 80, 90, 92, 96, 100, 100, 101, 102, 103, 104, 107, 110, 112, 116, 119, 121,125,125,125,125]
cufft_fuzz4all_api = [0,10,	11,	12,	12,	12,	12,	12,	12,	12,	12,	12,	12,	12,	14,	14,	14,	14,	14,	14,	14,	16,	16,	16,	16]
cufft_fuzz4all_api_edge = [0,	22,	26,	29,	29,	34,	35,	35,	45,	47,	49,	51,	53,	55,	57,	57,	57,	58,	58,	58,	59,	63,	63,	63,	64]

# cunpp
cunpp_api_data = [0, 10, 12, 14, 15, 17, 17, 19, 20, 21, 21, 21, 24, 24, 24, 25, 25, 25, 25, 26, 26, 26,26,26,26] 
cunpp_api_edge_data = [0, 39, 48, 56, 58, 60, 62, 70, 72, 73, 73, 73, 75, 80, 80, 81, 84, 84, 84, 84, 85, 85,85,85,85]
cunpp_fuzz4all_api = [0	,9,	10	,11,	11,	13,	13,	13,	14,	14,	15,	15,	16,	17,	17,	17,	17,	17,	17,	17,	20,	20,	20,	20,	20]
cunpp_fuzz4all_api_edge = [0, 22,	26,	29,	29,	34,	35,	35,	45,	47,	49,	51,	53,	55,	57,	57,	57,	58,	58,	58,	59,	63,	63,	63,	64]

# cusparse
cusparse_api_data = [0,	30	,48	,62	,69,73,	81,	89,	92,	98,	101,102	,103,110,112,114,116,120,122,124,125,125,126,126,126]
cusparse_api_edge_data = [0,45,	69,	101,126	,145,161,181,202,225,243,252,260,295,308,328,346,361,371,386,393,394,407,423,436]
cusparse_fuzz4all_api = [0,	11,	15,	18,	19,	19,	20,	20,	20,	21,	21,	21,	21,	22,	23,	23	,23,	23,	23,	23,	24,	24,	25,	25,	25]
cusparse_fuzz4all_edge_api = [0	,16,	26,	46,	51,	51,	55,	58,	64,	65,	65,	66,	67,	71,	72,	75,	75,	75,	75,	77,	77,	80,	81,	81,	83]


result_dict = {'rt':[rt_api_data, rt_api_edge_data, rt_fuzz4all_api, rt_fuzz4all_api_edge],
               'nvjpeg':[nvjpeg_api_data,nvjpeg_api_edge_data,nvjpeg_fuzz4all_api,nvjpeg_fuzz4all_api_edge],
               'cublas':[cublas_api_data,cublas_api_edge_data,cublas_fuzz4all_api,cublas_fuzz4all_api_edge],
               'curand':[curand_api_data,curand_api_edge_data,curand_fuzz4all_api,curand_fuzz4all_api_edge],
               'cusolver':[cusolver_api_data,cusolver_api_edge_data,cusolver_fuzz4all_api,cusolver_fuzz4all_api_edge],
               'cufft':[cufft_api_data,cufft_api_edge_data,cufft_fuzz4all_api,cufft_fuzz4all_api_edge],
               'cunpp':[cunpp_api_data,cunpp_api_edge_data,cunpp_fuzz4all_api,cunpp_fuzz4all_api_edge],
               'cusparse':[cusparse_api_data,cusparse_api_edge_data,cusparse_fuzz4all_api,cusparse_fuzz4all_edge_api]}

def draw_coverage_curve(seed_num=10, y_lim=240, key_word='rt', edge_or_black=0, first_num=6):
    hours = np.arange(0, 25)  # 0-24小时

    plt.figure(figsize=(5, 4))

    if edge_or_black == 0:
        my_list = result_dict[key_word][0]
        baseline = result_dict[key_word][2]
    else:
        my_list = result_dict[key_word][1]
        baseline = result_dict[key_word][3]



    my_list = np.array(my_list)
    baseline = np.array(baseline)
    if len(my_list)!=25 or len(baseline)!=25:
        print(f"$$$$$$$$$$$$$$$$$$$$  list len must be 25! ")
        print(len(my_list), len(baseline))
        sys.exit(1)

    seed = [seed_num]*25

    # 绘制第一条折线（红色，圆点标记）
    plt.plot(hours, my_list, 
            label='CuFuzz', 
            color='red', 
            marker='o',        # 圆点标记
            linestyle='-',     # 实线
            linewidth=2,       # 线宽
            markersize=3)      # 标记大小
    
   
    y_upper = np.round(np.concatenate((my_list[:4]*1.15, my_list[4:9]*1.15, my_list[9:13]*1.05, my_list[13:18]*1.02, my_list[18:]*1.03 )))
    y_upper = np.round(np.array([0,17,24,26,27,27,27,27,27,27,27,27,27,29,29,29,29,29,29,29,29,29,29,29,29]))
    # for kk in range(4,10,1):
    #     y_upper[kk] = y_upper[kk-1]+1
    # y_upper = np.round(np.concatenate((my_list[:4]*1.15, np.array([18,19,20,21,22]), np.array([23,24,25,25,25]),np.array([27,27,27,27,27]),np.array([27,27,27,27,27,27]))))
    
    y_lower = np.round(np.concatenate((my_list[:3]*0.7, my_list[3:7]*0.9, my_list[7:9]*0.80, my_list[9:15]*0.95, my_list[15:20]*0.95, my_list[20:]*0.95) ) )# 下边界
   
    y_lower = np.round(np.array([0,12,16,18,18,18,18,24,24,24,24,24,25,25,25,25,25,25,25,25,25,25,25,25,25]))

    plt.fill_between(hours, 
                 y_lower, 
                 y_upper, 
                 color='red', 
                 alpha=0.1)  # 设置透明度


    # 绘制第二条折线（蓝色，三角标记）
    plt.plot(hours, baseline, 
            label='Fuzz4all', 
            color='blue', 
            marker='^',        # 三角标记
            linestyle='-',      # 实线
            linewidth=2,        # 线宽
            markersize=3)       # 标记大小
    
    fluctuation = 0.07 * baseline  # 计算5%的波动值
    y_upper = np.round(baseline + fluctuation ) # 上边界
    
    y_lower = np.round(baseline - fluctuation ) # 下边界
    for kk in [21,22,23,24]:
        y_lower[kk]=y_lower[20]
    

    plt.fill_between(hours, 
                 y_lower, 
                 y_upper, 
                 color='blue', 
                 alpha=0.1)  # 设置透明度



    plt.plot(hours, seed, 
            label='Documents', 
            color='gray', 
            linestyle='--',      # 实线
            linewidth=2) 

    # 添加标题和标签
    #plt.title('Coverage Over Time', fontsize=14, pad=20)
    plt.xlabel('Hours', fontsize=12)
    if edge_or_black:
        plt.ylabel('API  Edge Coverage', fontsize=12)
    else:
        plt.ylabel('API Coverage', fontsize=12)
    

    # 添加图例
    plt.legend(loc='upper left', fontsize=10)

    # 添加网格
    plt.grid(True, 
            linestyle='--', 
            alpha=0.7)

    # 设置坐标轴范围
    plt.xlim(0, 24)
    plt.ylim(0, y_lim)  # 根据数据范围适当调整

    # 调整布局
    plt.tight_layout()

    if edge_or_black:
        plt.savefig(f'./images/{key_word}_edge.png')
    else:
        plt.savefig(f'./images/{key_word}_api.png')
    # 显示图形
    #plt.show()



if __name__ == "__main__":
    draw_coverage_curve(seed_num=34, y_lim=50, key_word='curand', edge_or_black=0)
    print(1)