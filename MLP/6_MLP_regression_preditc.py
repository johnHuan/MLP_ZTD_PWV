# -*- coding: utf-8 -*-
# @Time    : 2022/8/7 17:42
# @Author  : Zhang Huan
# @Email   : johnhuan@whu.edu.cn
# QQ       : 248404941
# @File    : 5_MLP_regression_train.py

import os

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
import joblib

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

if __name__ == '__main__':
    base_dir = 'D:/doctor/陆态网转换参数/data/train_test/'
    df_test = pd.read_csv(base_dir + 'test.csv', header=None)

    data_x = df_test.iloc[:, :-1].values
    data_y = df_test.iloc[:, -1].values

    # 对训练集每列的x的每一列分别进行归一化
    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    data_x_scalered = x_scaler.fit_transform(data_x)
    data_y_scalered = y_scaler.fit_transform(data_y.reshape(-1, 1))

    x_test_scalered = data_x_scalered[:, :]
    y_test_scalered = data_y_scalered[:, -1]  # 测试集中的输入参数 x, 训练集中的输出参数 y

    y_test_scalered_reshaped = y_test_scalered.reshape(-1, 1)
    y_test_inverse_scaler = y_scaler.inverse_transform(y_test_scalered_reshaped)

    mlp = joblib.load('mlp_model.pkl')

    # score of the model
    print(mlp.score(x_test_scalered, y_test_scalered))

    # test the model
    pred_scalered = mlp.predict(x_test_scalered)

    pred_scalered_reshaped = pred_scalered.reshape(-1, 1)

    pred_inverse_scaler = y_scaler.inverse_transform(pred_scalered_reshaped)

    print(mean_squared_error(y_test_scalered_reshaped, pred_scalered_reshaped))
    print(mean_squared_error(y_test_inverse_scaler, pred_inverse_scaler))

    result = pd.DataFrame({
        'pred': pred_inverse_scaler.reshape(-1),
        'test': y_test_inverse_scaler.reshape(-1),
        'pred_scalerd': pred_scalered_reshaped.reshape(-1),
        'test_scalerd': y_test_scalered_reshaped.reshape(-1)
    })
    result.to_csv(base_dir + 'outer_test_result.csv')

    plt.plot(pred_scalered_reshaped, label='pred')
    plt.plot(y_test_scalered_reshaped, label='test')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('MLP_ predictions')
    plt.legend()
    plt.savefig('outer_MLP_.png')
    plt.close()
    print(123)
