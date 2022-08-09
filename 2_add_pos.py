# -*- coding: utf-8 -*-
# @Time    : 2022/6/11 21:38
# @Author  : Zhang Huan
# @Email   : johnhuan@whu.edu.cn
# QQ       : 248404941
# @File    : 2_add_pos.py

import os

import numpy as np
import pandas as pd

if __name__ == '__main__':
    # todo 将经纬度和高程添加到ZTD与PWV数据中
    base_dir = 'D:/doctor/陆态网转换参数/'
    data_dir = 'data/without_pos/'
    # data_dir = 'data/f_2015_without_pos_2010_2014/'
    # data_dir = 'data/f_2015_predict_without_pos_2015/'
    pos_dir = 'GPW_PWV_CHINA/site.pos'
    pos_path = base_dir + pos_dir
    np_pos = np.loadtxt(pos_path, skiprows=1, dtype='str')
    np_pos_df = pd.DataFrame(np_pos)
    data_path = base_dir + data_dir
    stations = os.listdir(data_path)
    for station in stations:
        station_name = station[:-4]
        station_pos_names = np_pos_df[np_pos_df[0] == station_name]
        station_path = base_dir + data_dir + station
        print(station_path + '==========================')
        station_df_without_pos = pd.read_csv(station_path, header=None)
        station_pos = station_pos_names.append([station_pos_names] * (station_df_without_pos.shape[0] - 1))
        # todo 将 station_df_without_pos 和 station_pos合并
        station_df_without_pos.insert(3, 'l', station_pos.values[:, 1])
        station_df_without_pos.insert(4, 'b', station_pos.values[:, 2])
        station_df_without_pos.insert(5, 'h', station_pos.values[:, 3])
        # station_df_without_pos['l'] = station_pos.values[:, 1]
        # station_df_without_pos['b'] = station_pos.values[:, 2]
        # station_df_without_pos['h'] = station_pos.values[:, 3]
        station_df_without_pos.columns = ['YearDoy', 'year', 'doy', 'L', 'B', 'H', 'ZTD', 'PWV']
        station_df_without_pos.to_csv(base_dir + 'data/each_station_with_pos/' + station)
        # station_df_without_pos.to_csv(base_dir + 'data/2015_each_station_with_pos/' + station)
        # station_df_without_pos.to_csv(base_dir + 'data/f_2015_predict_each_station_with_pos_2015/' + station)
