
# 设定范围的函数
def set_bounds(value):
    return [value * (1 - 0.5), value * (1 + 0.5)]

def init_OTA_two():
    # 饱和区搜索值(单位微米)
    cap = 3.5e-12
    k1 = 1.8
    k2 = 8
    L1 = 2e-6
    L2 = 2e-6
    L3 = 1e-6
    L4 = 0.6e-6
    L5 = 0.6e-6
    r = 3000
    gmid1 = 12
    gmid2 = 12
    gmid3 = 10
    gmid4 = 10
    gmid5 =  10

    # 电容
    cap_min, cap_max = set_bounds(cap)
    k1_min, k1_max =set_bounds(k1)
    k2_min, k2_max =set_bounds(k2)
    
    L1_min, L1_max = set_bounds(L1)
    W1_min, W1_max = set_bounds(gmid1)

    
    L2_min, L2_max = set_bounds(L2)
    W2_min, W2_max = set_bounds(gmid2)

    
    L3_min, L3_max = set_bounds(L3)
    W3_min, W3_max = set_bounds(gmid3)

    
    L4_min, L4_max = set_bounds(L4)
    W4_min, W4_max = set_bounds(gmid4)

    # 管子五
    L5_min, L5_max = set_bounds(L5)
    W5_min, W5_max = set_bounds(gmid5)

    R_min, R_max = set_bounds(r)

    # 设置问题字典
    param_ranges = [
        (cap_min, cap_max),
        (k1_min, k1_max),
        (k2_min, k2_max),
        (L1_min, L1_max),
        (L2_min, L2_max),
        (L3_min, L3_max),
        (L4_min, L4_max),
        (L5_min, L5_max),
        (R_min, R_max),
        (W1_min, W1_max),
        (W2_min, W2_max),
        (W3_min, W3_max),
        (W4_min, W4_max),
        (W5_min, W5_max)
    ]
    # 定义一个包含所有约束的字典
    thresholds = {
        'gain': 60,
        'i_multiplier': 1.8,
        'i': 1e-3,
        'phase': 60,
        'gbw': 4e6
    }

    # 用于保存初始满足饱和区的值
    valid_x = dbx_alter = [[3.5e-12, 1.8, 8, 2e-6, 2e-6, 1e-6, 0.6e-6, 0.6e-6, 3000, 12, 12, 10, 10, 10]]
    valid_y = dby_alter = [[77.64, 2.345e-4, 67.2136, 38e6]]

    # 用于记录上一次满足条件的值
    last_valid_x = [3.5e-12, 1.8, 8, 2e-6, 2e-6, 1e-6, 0.6e-6, 0.6e-6, 3000, 12, 12, 10, 10, 10]
    last_valid_y = [77.64, 2.345e-4, 67.2136, 38e6]


    return param_ranges, thresholds, valid_x, valid_y, dbx_alter, dby_alter, last_valid_x, last_valid_y

