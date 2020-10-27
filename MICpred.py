#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from optparse import OptionParser

import pandas as pd

from predict.MLfunctions import LR
from predict.MLfunctions import RF
from predict.MLfunctions import GB
from predict.MLfunctions import DL

def predict_LR(func, col, ant_name, col_n, input_file, output):
    parser = OptionParser()
    parser.add_option("-p", "--path", type="string", dest="path", default=os.getcwd())
    parser.add_option("-i", "--input", type="string", dest="input", default=input_file)
    parser.add_option("-g", "--penalty", type="float", dest="Cval", default=1.0)
    parser.add_option("-r", "--ratio", type="float", dest="split", default=0.2)
    parser.add_option("-t", "--function", type="int", dest="function", default=func)
    (options, args) = parser.parse_args()

    lr = LR(options.path).LogisticRegression(options.input,options.split,options.Cval,options.function, col, ant_name, output, col_n)
    return lr

def predict_RF(func, col, ant_name, col_n, input_file, output):
    parser = OptionParser()
    parser.add_option("-p", "--path", type="string", dest="path", default=os.getcwd())
    parser.add_option("-i", "--input", type="string", dest="input", default=input_file)
    parser.add_option("-r", "--ratio", type="float", dest="split", default=0.2)
    
    parser.add_option("-f", "--maxfeatures", type="string", dest="maxf", default=None)
    parser.add_option("-n", "--nestimators", type="float", dest="numest", default=100)
    parser.add_option("-t", "--function", type="int", dest="function", default=func)
   
    (options, args) = parser.parse_args()
    print(options.maxf)

    rf = RF(options.path).RandomForests(options.input, options.split,options.maxf,"2000",options.function, col, ant_name, output, col_n)
    return rf

def predict_GB(func, col, ant_name, col_n, input_file, output):
    parser = OptionParser()
    parser.add_option("-p", "--path", type="string", dest="path", default=os.getcwd())
    parser.add_option("-i", "--input", type="string", dest="input", default=input_file)
    parser.add_option("-r", "--ratio", type="float", dest="split", default=0.2)
    
    parser.add_option("-f", "--maxfeatures", type="string", dest="maxf", default='sqrt')
    parser.add_option("-n", "--nestimators", type="float", dest="numest", default=300)
    parser.add_option("-t", "--function", type="int", dest="function", default=func)
   
    (options, args) = parser.parse_args()

    # gb = GB(options.path).GradientBoosting(options.input, options.split,options.maxf,options.numest,options.function, col, ant_name, output, col_n)
    xgb = GB(options.path).XGBoosting(options.input, options.split, options.maxf, options.numest,options.function, col, ant_name, output, col_n)
    return xgb

def predict_DL(func, col, ant_name, col_n, input_file, output):
    parser = OptionParser()
    parser.add_option("-p", "--path", type="string", dest="path", default=os.getcwd())
    parser.add_option("-i", "--input", type="string", dest="input", default=input_file)
    parser.add_option("-r", "--ratio", type="float", dest="split", default=0.2)
    
    parser.add_option("-d", "--drop_out", type="float", dest="dropout", default=0.2)
    parser.add_option("-n", "--firstlayer", type="string", dest="firstlayer", default=300)
    parser.add_option("-m", "--interlayer", type="string", dest="interlayer", default=150)
    parser.add_option("-l", "--layer", type="string", dest="layer", default=4)
    parser.add_option("-t", "--function", type="int", dest="function", default=func)
    
    (options, args) = parser.parse_args()

    # if func == 1:
    #     lstm_ac = DL(options.path).DL_lstm(options.input, options.split, options.function, col, ant_name, output, col_n)
    # else:
    #     lstm_ac = DL(options.path).DL_lstm_calssifier(options.input, options.split, options.function, col, ant_name, output, col_n)

    fcn_ac = DL(options.path).DeepLearning_fcn(options.input, options.split, int(options.firstlayer), int(options.interlayer),options.dropout, int(options.layer),options.function, col, ant_name, output, col_n)
    return fcn_ac#, lstm_ac#, fcn_ac

def mkdir(path):
    # 引入模块
    import os
    # 去除首位空格
    path = path.strip()
    # 去除尾部 \ 符号
    path = path.rstrip("\\")
    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists = os.path.exists(path)
    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(path)
        print(path + ' 创建成功')
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print(path + ' 目录已存在')
        return False
    
def main():
    cols = [0,1,2,3,5,6,8,9,11,13,14]
    ant_names = ['AMP', 'AUG', 'CIP', 'AXO', 'CHL', 'COT', 'FOX', 'GEN', 'NAL', 'TET', 'TIO']
    feature_number = [10,20,30,40,50,100,200,500,1000]
    # col_n = 500

    for i in range(0,1):
        func = 0
        summary = []
        for j in range(0,11):
            ant_name = ant_names[j]
            col = cols[j]
            input_file = 'test_data/snp_mic.csv'    #mic_liner_new #genes_copynum_mic.csv' #'+ant_name+'_1000_arg_mic_liner.csv' #genes_rm_arg_mic #DeepARG_copynumber_plus_potential_arg_table
            output = '/results_snp_mic_3/' + ant_name + str(func) + '/'
            mkdir(os.getcwd() + output)
            print(func, col, ant_name, feature_number[i], input_file, output)
            # LR_ac = predict_LR(func, col, ant_name, col_n, input_file, output)
            # predict_RF(func, col, ant_name, col_n, input_file, output)
            XGB_ac = predict_GB(func, col, ant_name, feature_number[i], input_file, output)
            # FCN_ac = predict_DL(func, col, ant_name, feature_number[i], input_file, output)
        #     summary.append([ant_name, feature_number[i], XGB_ac, FCN_ac])#LR_ac, GB_ac, XGB_ac, FCN_ac, lstm_ac
        # print(summary)
        # summary = pd.DataFrame(summary, columns=['Antibiotic', 'feature_number ', 'XGB_ac',
        #                                          'FCN_ac'])  # 'LR_ac', 'GB_ac', 'XGB_ac', 'FCN_ac', 'LSTM_ac'
        # summary.to_csv(os.getcwd() + '/results_snp_mic_' + str(feature_number[i]) + '/' + 'summary.csv', sep='\t')
    # summary = list(zip(y_test, y_pred, y_c))


if __name__ == '__main__':
    main()
