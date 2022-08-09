# -*- coding: utf-8 -*-
# @Time    : 2022/6/17 17:32
# @Author  : Zhang Huan
# @Email   : johnhuan@whu.edu.cn
# QQ       : 248404941
# @File    : tell_train_test.py
import os

import pandas as pd


def print_hi(name):
    print(f'Hi, {name}')


if __name__ == '__main__':
    base_dir = 'D:/doctor/陆态网转换参数/'
    data_path = base_dir + 'data/each_station_with_pos/'
    pos_path = base_dir + 'code/python/MLP/40_col_stations.xlsx'
    pos = pd.read_excel(pos_path)
    stations = os.listdir(data_path)
    for station in stations:
        station_path = data_path + station
        station_name = station[:-4]

        # 首先全部存一遍
        station_df_without_pos = pd.read_csv(station_path)
        path = base_dir + 'data/train_test/'
        station_df_without_pos.to_csv(path + 'dataset.csv', mode='a', header=None, index=None)
        if station_name in pos['name_CMONOC'].values:
            print(station + ' is a test station-----------------------------------------------')
            # 测试集
            station_df_without_pos = pd.read_csv(station_path)
            station_df_without_pos.to_csv(path + 'test.csv', mode='a', header=None, index=None)
        else:
            print(station + ' is a train station *********************************************')
            # 训练集
            station_df_without_pos = pd.read_csv(station_path)
            station_df_without_pos.to_csv(path + 'train.csv', mode='a', header=None, index=None)

    print_hi('ZhangHuan')
