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
    df_train = pd.read_csv(base_dir + 'train.csv', header=None)
    df_test = pd.read_csv(base_dir + 'test.csv', header=None)
    data = pd.read_csv(base_dir + 'dataset.csv', header=None)
    data = data.iloc[:, 1:]
    # 需要将 训练集合测试集放在一起进行归一化，所以就需要将两个文件拼接起来
    flag_train, flag_test = df_train.shape[0], df_test.shape[0]

    data_x = data.iloc[:, :-1].values
    data_y = data.iloc[:, -1].values
    # 对训练集每列的x的每一列分别进行归一化
    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    data_x_scalered = x_scaler.fit_transform(data_x)
    data_y_scalered = y_scaler.fit_transform(data_y.reshape(-1, 1))

    x_train_scalered = data_x_scalered[:df_train.shape[0], :]
    y_train_scalered = data_y_scalered[:df_train.shape[0], -1]  # 训练集中的输入参数 x, 训练集中的输出参数 y
    x_test_scalered = data_x_scalered[df_train.shape[0]:, :]
    y_test_scalered = data_y_scalered[df_train.shape[0]:, -1]  # 测试集中的输入参数 x, 训练集中的输出参数 y

    y_test_scalered_reshaped = y_test_scalered.reshape(-1, 1)
    y_test_inverse_scaler = y_scaler.inverse_transform(y_test_scalered_reshaped)

    mlp = MLPRegressor(hidden_layer_sizes=(900, 6), activation='relu', solver='adam', shuffle=True,
                       early_stopping=True, verbose=True,
                       learning_rate_init=0.001, alpha=0.001,
                       batch_size=13, learning_rate='adaptive',
                       max_iter=1000, random_state=5, tol=0.001,
                       validation_fraction=0.2
                       )
    joblib.dump(mlp, './MLP_model.pkl')  # 模型保存
    # train the model
    mlp.fit(x_train_scalered, y_train_scalered)
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
    result.to_csv(base_dir + 'result.csv')

    plt.plot(pred_scalered_reshaped, label='pred')
    plt.plot(y_test_scalered_reshaped, label='test')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('MLP_ predictions')
    plt.legend()
    plt.savefig('./MLP_.png')
    plt.close()
    print(123)
