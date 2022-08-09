# -*- coding: utf-8 -*-
# @Time    : 2022/6/11 21:39
# @Author  : Zhang Huan
# @Email   : johnhuan@whu.edu.cn
# QQ       : 248404941
# @File    : 1_data_normalize.py

import os

import numpy as np
import pandas as pd

if __name__ == '__main__':
    base_dir = 'D:/doctor/陆态网转换参数/'
    source_path = 'GPW_PWV_CHINA/'
    target_dir = 'data/without_pos/'
    # target_dir = 'data/single_station/'
    # stations = ['ahbbpp_']
    # years = list(range(2010, 2016))
    years = list(range(2005, 2016))
    for year in years:
        path_year = base_dir + source_path + str(year) + '/'
        stations = os.listdir(path_year)
        print(path_year + '------------------------------')
        for station in stations:
            station_name = station[0:4]
            # path_station = path_year + station + str(year) + '.plt'
            path_station = path_year + station  # + str(year) + '.plt'
            print(path_station + '*************************************************************')
            np_station = np.loadtxt(path_station)
            year_add = np_station[:, 0] + year * 1000
            year_vector = np_station[:, 0] + year - np_station[:, 0]
            df = pd.DataFrame(np_station)
            df.insert(0, 'year_doy', year_add)
            df.insert(1, 'year', year_vector)
            column_name = ['year_doy', 'year', 'doy', 'PWV', 'PWV_Err', 'ZTD', 'P', 'T1', 'T2']
            df.columns = column_name
            target_df = df[['year_doy', 'year', 'doy', 'ZTD', 'PWV']]
            target_path = base_dir + target_dir + station_name + '.csv'
            target_df.to_csv(target_path, mode='a', header=None, index=None)
