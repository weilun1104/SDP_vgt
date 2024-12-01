# -*- coding: utf-8 -*-
import sys
import os
pythonpath = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0,pythonpath)
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import numpy as np
import pandas as pd
from utils.util import seed_set
from benchmarks.synthetic_functions import Hartmann6, Griewank,Ackley,RosenBrock,Rastrigin,Michalewicz,Levy, OTA_three
from Experiment.exp_under_different_train_sample.config_OTA_three import init_OTA_three
from Model.Point_search.CONBO import plot
from VGT import VGT


# if os.path.exists('result.csv'):
#     os.remove('result.csv')

def save_data(best_all_y, iter_times, save_path):
    best_all_y1 = [item[0][0].item() for item in best_all_y]
    best_all_y2 = [item[0][1].item() for item in best_all_y]
    best_all_y3 = [item[0][2].item() for item in best_all_y]
    best_all_y4 = [item[0][3].item() for item in best_all_y]
    # 将5个list保存到一个CSV文件中
    df = pd.DataFrame({
        'iter_times': iter_times,
        'gain(db)': best_all_y1,
        'dc_current': best_all_y2,
        'phase': best_all_y3,
        'GBW(MHZ)': best_all_y4,
    })
    df.to_csv(save_path, index=False)

if __name__=='__main__':
    for m in range(300, 310):
        seed = m
        seed_set(seed)
        param_ranges, thresholds, valid_x, valid_y, dbx_alter, dby_alter, last_valid_x, last_valid_y = init_OTA_three()
        # dims = 5
        # lb = np.zeros(dims)
        # ub = np.ones(dims)
        lb = np.array([min_range for min_range, _ in param_ranges])
        ub = np.array([max_range for _, max_range in param_ranges])
        lb = np.log(lb)
        ub = np.log(ub)
        f = OTA_three 

        n_init = 20
        max_iter = 400
        #N_neighbor = 80
        Cp = 150#0.2
        num_samples=40

        use_approximation = True #False
        N_neighbor = 20

        best_all_y = []
        best_y = None
        min_current = float('inf')
        dbx_alter = np.log(dbx_alter)
        agent = VGT(dbx_alter,min_current,best_all_y,best_y, f,lb,ub,n_init,max_iter,Cp= Cp,use_approximation = use_approximation,N_neighbor=N_neighbor, num_samples = num_samples)
        best_all_y = agent.search()

        # 实验结果保存路径  
        file_path = ('C:/DAC/VGT-main/Experiment/exp_under_different_train_sample/exp_design_1/baseline_Model_v1/vgt/vgt_OTA_three_seed_{}.csv').format(m)
        # 实验结果计算路径
        cal_path = (
                    "C:\\DAC\\VGT-main\\Experiment\\exp_under_different_train_sample\\exp_design_1\\baseline_Model_v1\\vgt\\vgt_OTA_three_seed_")
        # 均值方差计算结果保存路径
        to_path = (
                'C:\\DAC\\VGT-main\\Experiment\\exp_under_different_train_sample\\exp_design_1_report\\vgt_OTA_three_current_mean_var_strand.csv')

        # iter_times = list(range(1, len(best_all_y) + 1))
        # save_data(best_all_y, iter_times, file_path)
        plot(cal_path, to_path, [100, 101, 102, 200, 300])
