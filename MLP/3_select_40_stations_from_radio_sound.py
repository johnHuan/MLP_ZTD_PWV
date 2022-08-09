# -*- coding: utf-8 -*-
# @Time    : 2022/8/8 20:03
# @Author  : Zhang Huan
# @Email   : johnhuan@whu.edu.cn
# QQ       : 248404941
# @File    : 3_select_40_stations_from_radio_sound.py
import os

import pandas as pd

if __name__ == '__main__':
    """
    1. 从探空站的目录下拿到40个有数据的站名，作为与陆态网的共址站 
        D:/doctor/陆态网转换参数/code/matlab/探空站/radio_sound_txt/
    2. 从51个共址站中找到40个有数据的探空站和陆态网站，作为有效的共址站，将剩下11个探空站没有数据的站 对应的陆态网站原放回到训练集中参与MLP训练
    """
    radio_sound_dir = 'D:/doctor/陆态网转换参数/code/matlab/探空站/radio_sound_txt/'
    陆态网和探空站并址带坐标的文件名 = './stations_col_CMONOC.xlsx'
    df = pd.read_excel(陆态网和探空站并址带坐标的文件名, index_col=0)
    stations = os.listdir(radio_sound_dir)
    col_stations_df = pd.DataFrame(columns=df.columns)
    for station in stations:
        station_name = station[0:11]  # CHM00050527-drvd ->  CHM00050527
        row = df.loc[df['name_radio_sound'] == station_name]
        col_stations_df = col_stations_df.append(row, ignore_index=True)
    col_stations_df.to_excel('40_col_stations.xlsx')
