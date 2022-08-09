# -*- coding: utf-8 -*-
# @Time    : 2022/8/8 21:36
# @Author  : Zhang Huan
# @Email   : johnhuan@whu.edu.cn
# QQ       : 248404941
# @File    : evaluate_MLP.py


import os

import numpy as np
import pandas as pd

if __name__ == '__main__':
    base_dir = 'D:/doctor/陆态网转换参数/data/train_test/'
    # target_filename = base_dir + 'MLP_accuracy.xlsx'
    result_file = base_dir + 'result.csv'

    df_result = pd.read_csv(result_file)

    test_df = df_result['test']
    pred_df = df_result['pred']

    Ai = test_df - pred_df
    Ai_pred = pred_df - np.mean(pred_df)
    Ai_true = test_df - np.mean(test_df)
    bias = np.mean(Ai)
    std = np.sqrt(np.mean((Ai - bias) ** 2))
    mae = np.mean(np.abs(Ai))
    r2 = np.sum(Ai_pred * Ai_true) / np.sqrt(np.sum(Ai_pred ** 2) * np.sum(Ai_true ** 2))
    rms = np.sqrt(np.mean(Ai ** 2))
    print("bias: %f\t std: %f\t MAE: %f \t R: %f \t RMSE: %f" % (bias, std, mae, r2, rms))
    # target_df = pd.DataFrame({
    #     'bias': bias,
    #     'std': std,
    #     'mae': mae,
    #     'R2': r2,
    #     'rms': rms
    # })
    # target_df.to_excel(target_filename)
