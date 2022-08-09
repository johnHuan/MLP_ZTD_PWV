# -*- coding: utf-8 -*-
# @Time    : 2022/6/17 15:37
# @Author  : Zhang Huan
# @Email   : johnhuan@whu.edu.cn
# QQ       : 248404941
# @File    : find266_stations.py
import os

import numpy as np
import pandas as pd

if __name__ == '__main__':
    # todo 将经纬度和高程添加到ZTD与PWV数据中
    base_dir = 'D:/doctor/陆态网转换参数/'
    data_dir = 'data/without_pos/'
    pos_dir = 'GPW_PWV_CHINA/site.pos'
    pos_path = base_dir + pos_dir
    np_pos = np.loadtxt(pos_path, skiprows=1, dtype='str')
    np_pos_df = pd.DataFrame(np_pos)
    data_path = base_dir + data_dir
    stations = os.listdir(data_path)
    column_names = ['name', 'lat', 'lon', 'height']
    np_pos_df.columns = column_names
    # todo 从np_pos_df中祛除不含在stations中的站，然后将其保存下来
    station_names = []
    for station in stations:
        station_name = station[:-4]
        station_names.append(station_name)
    sn_df = pd.DataFrame({'name': station_names})
    # todo sn_df 和 np_pos_df取交集
    target_df = pd.merge(np_pos_df, sn_df, on=['name', 'name'])
    target_df.to_excel(base_dir + 'data/266_stations.xlsx')
