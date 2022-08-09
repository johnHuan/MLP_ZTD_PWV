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
    validate_stations = ["lnsy", "jlcl", "hrbn", "bjfs", "hely", "sxlq", "xjds", "gsjy", "gslz", "xzar", "xznq", "xzge",
                         "tian", "ahbb", "jsly", "wuhn", "haqs", "hbxf", "jxja", "gxwz", "xiam", "xiag", "ynth", "yngm"]
    base_dir = 'D:/doctor/陆态网转换参数/data/'
    data_path = base_dir + '2015_each_station_with_pos/'
    stations = os.listdir(data_path)
    for station in stations:
        station_path = data_path + station
        station_name = station[:-4]
        if validate_stations.__contains__(station_name):
            # 验证集
            station_df_without_pos = pd.read_csv(station_path)
            # station_df_without_pos.to_csv(base_dir + 'train_test/test.csv', mode='a', header=None, index=None)
            station_df_without_pos.to_csv(base_dir + '2015_train_test/test.csv', mode='a', header=None, index=None)
        else:
            # 训练集
            station_df_without_pos = pd.read_csv(station_path)
            # station_df_without_pos.to_csv(base_dir + 'train_test/train.csv', mode='a', header=None, index=None)
            station_df_without_pos.to_csv(base_dir + '2015_train_test/train.csv', mode='a', header=None, index=None)

    print_hi('ZhangHuan')
