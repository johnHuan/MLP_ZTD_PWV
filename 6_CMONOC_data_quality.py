# -*- coding: utf-8 -*-
# @Time    : 2022/8/8 22:03
# @Author  : Zhang Huan
# @Email   : johnhuan@whu.edu.cn
# QQ       : 248404941
# @File    : 6_CMONOC_data_quality.py
import os

import pandas as pd

if __name__ == '__main__':
    base_dir = 'D:/doctor/陆态网转换参数/data/each_station_with_pos/'
    stations = os.listdir(base_dir)
    df = pd.DataFrame(columns=['name', 'b', 'l', 'data_len'])
    for station in stations:
        station_name = station[:-4]
        each_df = pd.read_csv(base_dir + station, usecols=['L', 'B', 'year'], index_col=None)
        length = len(each_df['year'].unique())
        df = df.append([{'name': station_name,
                         'l': each_df['L'][0],
                         'b': each_df['B'][0],
                         'data_len': length}])
    df.to_excel('D:/doctor/陆态网转换参数/data/station_data_len.xlsx')
