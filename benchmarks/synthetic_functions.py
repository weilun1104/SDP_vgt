# -*- coding: utf-8 -*-



import numpy as np
import time
import sys
import os
import torch
pythonpath = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0,pythonpath)
from Simulation.Data.lyngspice_master.lyngspice_master.examples.simulation_V3 import OTA_two_simulation_all
from Simulation.Data.lyngspice_master.lyngspice_master.examples.simulation_V4 import OTA_three_simulation_all
from Simulation.Data.lyngspice_master.lyngspice_master.examples.simulation_V5 import bandgap_all_simulation
from Simulation.Data.lyngspice_master.lyngspice_master.examples.simulation_ota2_tsmc40 import OTA_two_simulation_gmid_pro

#二级放大电路约束函数
def two_constraint_1(y):
    return (y[0, 0] > (60)).item()

# def two_constraint_2(y):
#     return (3e-4 < y[0, 1] < 4e-4).item()

def two_constraint_3(y):
    return (y[0, 2] > (60)).item()

def two_constraint_4(y):
    return (y[0, 3] > (4e6)).item()

def two_all_constraints(y):
    return two_constraint_1(y)  and two_constraint_3(y) and two_constraint_4(y)

# 定义约束函数
def three_constraint_1(y):
    return (y[0, 0] > (80)).item()

def three_constraint_3(y):
    return (y[0, 2] > (60)).item()

def three_constraint_4(y):
    return (y[0, 3] > (2e6)).item()

def three_all_constraints(y):
    return three_constraint_1(y) and three_constraint_3(y) and three_constraint_4(y)

# bandgap约束函数
# def bandgap_constraint_1(y):
#     return (y[0, 0] > (80)).item()

def bandgap_constraint_2(y):
    return (y[0, 1] < 5e-5).item()

def bandgap_constraint_3(y):
    return (y[0, 2] > 60).item()

def bandgap_all_constraints(y):
    return bandgap_constraint_2(y) and bandgap_constraint_3(y)

def bandgap(x, min_current, best_all_y, best_y):
    x_tensor = torch.tensor(x, dtype=torch.float32)  # 需要转换为 PyTorch tensor
    x_tensor = torch.exp(x_tensor).to('cuda:0')
    ytr = torch.empty((0,), dtype=torch.double).to('cuda:0')
    valid_x = torch.empty((0, x_tensor.shape[1]), dtype=torch.double).to('cuda:0')  # 初始化为空张量，形状匹配 x_tensor 的列数

    for x in x_tensor:
        x = x.unsqueeze(0)
        results = bandgap_all_simulation(x).to('cuda:0')
        ppm, dc_current, psrr = results[0]
        print(f"ppm: {ppm}, DC Current: {dc_current}, psrr: {psrr}")

        if bandgap_all_constraints(results):  # 只有满足约束条件的结果才会进入
            # 检查 valid_x 是否为空，如果为空则直接赋值，否则进行拼接
            if valid_x.numel() != 0:
                valid_x = torch.cat((valid_x, x), dim=0)  # 将满足条件的 x 添加到 valid_x 中
            else:
                valid_x = x  # 如果 valid_x 为空，直接赋值为 x

            if results[0][0].item() < min_current:
                min_current = results[0][0].item()
                best_y = results.clone()

            if ytr.numel() != 0: 
                ytr = torch.cat((ytr, results[0][0].unsqueeze(0)), dim=0)
            else:
                ytr = results[0][0].unsqueeze(0)
        
        best_all_y.append(best_y)


    valid_x = torch.log(valid_x)  # 对 valid_x 取对数
    return valid_x ,ytr.cpu().numpy(), min_current, best_all_y, best_y # 返回 valid_x

def OTA_three(x, min_current, best_all_y, best_y):
    x_tensor = torch.tensor(x, dtype=torch.float32)  # 需要转换为 PyTorch tensor
    x_tensor = torch.exp(x_tensor).to('cuda:0')
    ytr = torch.empty((0,), dtype=torch.double).to('cuda:0')
    valid_x = torch.empty((0, x_tensor.shape[1]), dtype=torch.double).to('cuda:0')  # 初始化为空张量，形状匹配 x_tensor 的列数

    for x in x_tensor:
        x = x.unsqueeze(0)
        results = OTA_three_simulation_all(x).to('cuda:0')
        gain, dc_current, phase, GBW = results[0]
        print(f"Gain: {gain}, DC Current: {dc_current}, Phase: {phase}, GBW: {GBW}")

        if three_all_constraints(results):  # 只有满足约束条件的结果才会进入
            # 检查 valid_x 是否为空，如果为空则直接赋值，否则进行拼接
            if valid_x.numel() != 0:
                valid_x = torch.cat((valid_x, x), dim=0)  # 将满足条件的 x 添加到 valid_x 中
            else:
                valid_x = x  # 如果 valid_x 为空，直接赋值为 x

            if results[0][1].item() < min_current:
                min_current = results[0][1].item()
                best_y = results.clone()

            if ytr.numel() != 0: 
                ytr = torch.cat((ytr, results[0][1].unsqueeze(0)), dim=0)
            else:
                ytr = results[0][1].unsqueeze(0)

        best_all_y.append(best_y)

    valid_x = torch.log(valid_x)  # 对 valid_x 取对数
    return valid_x ,ytr.cpu().numpy(), min_current, best_all_y, best_y # 返回 valid_x

def OTA_two(x, min_current, best_all_y, best_y):
    x_tensor = torch.tensor(x, dtype=torch.float32)  # 需要转换为 PyTorch tensor
    x_tensor = torch.exp(x_tensor).to('cuda:0')
    ytr = torch.empty((0,), dtype=torch.double).to('cuda:0')
    valid_x = torch.empty((0, x_tensor.shape[1]), dtype=torch.double).to('cuda:0')  # 初始化为空张量，形状匹配 x_tensor 的列数
    k = 0
    for x in x_tensor:
        k = k+1
        x = x.unsqueeze(0)
        results = OTA_two_simulation_gmid_pro(x).to('cuda:0')
        gain, dc_current, phase, GBW = results[0]
        print(f"Gain: {gain}, DC Current: {dc_current}, Phase: {phase}, GBW: {GBW}")

        if two_all_constraints(results):  # 只有满足约束条件的结果才会进入
            # 检查 valid_x 是否为空，如果为空则直接赋值，否则进行拼接
            if valid_x.numel() != 0:
                valid_x = torch.cat((valid_x, x), dim=0)  # 将满足条件的 x 添加到 valid_x 中
            else:
                valid_x = x  # 如果 valid_x 为空，直接赋值为 x

            if results[0][1].item() < min_current:
                min_current = results[0][1].item()
                best_y = results.clone()

            if ytr.numel() != 0: 
                ytr = torch.cat((ytr, results[0][1].unsqueeze(0)), dim=0)
            else:
                ytr = results[0][1].unsqueeze(0)
                
        best_all_y.append(best_y)
    valid_x = torch.log(valid_x)  # 对 valid_x 取对数
    return valid_x ,ytr.cpu().numpy(), min_current, best_all_y, best_y # 返回 valid_x

def Levy(x):
    xs = np.atleast_2d(x)*15-5
    dim = xs.shape[1]
    ws = 1 + (xs - 1.0) / 4.0
    val = np.array([np.sin(np.pi * w[0]) ** 2 + \
        np.sum((w[1:dim - 1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * w[1:dim - 1] + 1) ** 2)) + \
        (w[dim - 1] - 1) ** 2 * (1 + np.sin(2 * np.pi * w[dim - 1])**2) for w in ws])
        
    print('val = ',val)
    with open('result.csv','a+') as f:
        for s in val:
            f.write(str(time.time())+','+str(s)+','+'\n')
    return val

def Michalewicz(xin):
    xs = np.atleast_2d(xin)*np.pi
    dim = xs.shape[1]
    m = 10
    result = -np.sum( np.sin(xs) * np.sin((1+np.arange(dim)) * xs**2 / np.pi) ** (2 * m), axis=-1)

    #result = np.array([100*((x[1:]-x[:-1]**2)**2).sum() + ((x[:-1]-1)**2).sum()  for x in xs])
    print('val = ',result)
    
    with open('result.csv','a+') as f:
        for s in result:
            f.write(str(time.time())+','+str(s)+','+'\n')
    return result


def RosenBrock(xin):
    xs = np.atleast_2d(xin)*4.096-2.048
    result = np.array([100*((x[1:]-x[:-1]**2)**2).sum() + ((x[:-1]-1)**2).sum()  for x in xs])
    print('val = ',result)
    
    with open('result.csv','a+') as f:
        for s in result:
            f.write(str(time.time())+','+str(s)+','+'\n')
            
    
    return result

def Griewank(xin):
    xs = np.atleast_2d(xin)*1000-500
    result = np.array([(x**2).sum()/4000 + 1 - np.cos(x/np.sqrt(range(1,len(x)+1))).prod()  for x in xs])
    with open('result.csv','a+') as f:
        for s in result:
            f.write(str(time.time())+','+str(s)+','+'\n')
    print('val=',result)
    return result



def Schwefel(xin):
    xs = np.atleast_2d(xin)*1000-500
    val = 418.9829 * xs.shape[1] + np.array([(-x*np.sin(np.sqrt(np.abs(x)))).sum() for x in xs])
    with open('result.csv','a+') as f:
        for s in val:
            f.write(str(time.time())+','+str(s)+','+'\n')
    print('val = ',val)
    return val

def Rastrigin(xin):
    xs = np.atleast_2d(xin)*10.24-5.12
    sum_ = np.array([np.dot(x,x)+10*len(x)-10*np.sum(np.cos(2*np.pi*x)) for x in xs])
    with open('result.csv','a+') as f:
        for s in sum_:
            f.write(str(time.time())+','+str(s)+','+'\n')
    # y = np.array([sum_])
    print('yval=',sum_)
    return sum_



def Ackley(xin):
    xs = np.atleast_2d(xin)*15-5
    result = np.array([(-20*np.exp(-0.2 * np.sqrt(np.inner(x,x) / x.size )) -np.exp(np.cos(2*np.pi*x).sum() /x.size) + 20 +np.e ) for x in xs])
    print('val = ',result)
    with open('result.csv','a+') as f:
        for s in result:
            f.write(str(time.time())+','+str(s)+','+'\n')
    return result


def Hartmann6(xin):
    ALPHA = np.array([1.0, 1.2, 3.0, 3.2])

    A = np.array([
        [10, 3, 17, 3.5, 1.7, 8],
        [0.05, 10, 17, 0.1, 8, 14],
        [3, 3.5, 1.7, 10, 17, 8],
        [17, 8, 0.05, 10, 0.1, 14],
        ])

    P = 0.0001*np.array([
        [1312, 1696, 5569, 124, 8283, 5886],
        [2329, 4135, 8307, 3736, 1004, 9991],
        [2348, 1451, 3522, 2883, 3047, 6650],
        [4047, 8828, 8732, 5743, 1091, 381],
        ])
    
    xs = np.atleast_2d(xin)
    val = -np.array([(ALPHA*np.exp(-(A*((x-P)**2)).sum(axis=1))).sum() for x in xs])
    print('vals = ',val)
    with open('result.csv','a+') as f:
        for s in val:
            f.write(str(time.time())+','+str(s)+','+'\n')
    return val 



def Hartmann6_500(xin):
    ALPHA = np.array([1.0, 1.2, 3.0, 3.2])

    A = np.array([
        [10, 3, 17, 3.5, 1.7, 8],
        [0.05, 10, 17, 0.1, 8, 14],
        [3, 3.5, 1.7, 10, 17, 8],
        [17, 8, 0.05, 10, 0.1, 14],
        ])

    P = 0.0001*np.array([
        [1312, 1696, 5569, 124, 8283, 5886],
        [2329, 4135, 8307, 3736, 1004, 9991],
        [2348, 1451, 3522, 2883, 3047, 6650],
        [4047, 8828, 8732, 5743, 1091, 381],
        ])
    
    xs1 = np.atleast_2d(xin)
    xs = xs1[:,np.array([3,9,17,23,31,44])]
    val = -np.array([(ALPHA*np.exp(-(A*((x-P)**2)).sum(axis=1))).sum() for x in xs])
    print('vals = ',val)
    with open('result.csv','a+') as f:
        for s in val:
            f.write(str(time.time())+','+str(s)+','+'\n')
    return val 



def Ackley10_500(xin):
    xs1 = np.atleast_2d(xin)*15-5
    xs = xs1[:,np.array([3,9,17,23,31,44,233,157,324,412])]
    result = np.array([(-20*np.exp(-0.2 * np.sqrt(np.inner(x,x) / x.size )) -np.exp(np.cos(2*np.pi*x).sum() /x.size) + 20 +np.e ) for x in xs])
    print('val = ',result)
    with open('result.csv','a+') as f:
        for s in result:
            f.write(str(time.time())+','+str(s)+','+'\n')
    return result








