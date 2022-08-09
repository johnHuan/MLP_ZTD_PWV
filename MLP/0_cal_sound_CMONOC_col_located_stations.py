# -*- coding: utf-8 -*-
# @Time    : 2022/8/8 15:52
# @Author  : Zhang Huan
# @Email   : johnhuan@whu.edu.cn
# QQ       : 248404941
# @File    : 0_cal_sound_CMONOC_col_located_stations.py

import numpy as np
import pandas as pd
from numpy import sqrt


def cal_distance(lon, lat, df, threshold):
    lons, lats, station_names, distances = [], [], [], []
    for index, row in df.iterrows():
        lat_s = row['B']
        lon_s = row['L']
        distance = sqrt((lon_s - lon) ** 2 + (lat_s - lat) ** 2)
        print(distance)
        if distance < threshold:
            lons.append(lon_s)
            lats.append(lat_s)
            station_names.append(row['name'])
            distances.append(distance)
    if lons:
        # 此时找到小于阈值的站有可能不止一个，需要找出距离最小的站
        df = pd.DataFrame({'name': station_names, 'B': lats, 'L': lons, 'distances': distances})
        nearest_station = df.iloc[df['distances'].idxmin(), :]
        print("=================================有了==")
        return nearest_station
    return 0


if __name__ == '__main__':
    base_dir = 'D:/doctor/陆态网转换参数/'
    stations_radio_sound_path = base_dir + 'code/matlab/探空站/stations.xlsx'
    stations_CMONOC_path = base_dir + 'GPW_PWV_CHINA/stations.xlsx'
    df_radio_sound = pd.read_excel(stations_radio_sound_path)
    df_CMONOC = pd.read_excel(stations_CMONOC_path)
    radio_sound_lats, radio_sound_lons, radio_sound_station_names = [], [], []
    CMONOC_lats, CMONOC_lons, CMONOC_station_names = [], [], []
    distances = []
    for index, row in df_radio_sound.iterrows():
        lon_radio_sound = row['L']
        lat_radio_sound = row['B']
        name_radio_sound = row['name'].lstrip().rstrip()
        # 用探空站的经纬度去陆态网中找 小于阈值的站 返回陆态网的经纬度和测站
        nearest_station = cal_distance(lon_radio_sound, lat_radio_sound, df_CMONOC, 0.1)
        if nearest_station is not 0:  # 找到了含有共址站的 CMONOC站
            CMONOC_lon = nearest_station['L']
            CMONOC_lat = nearest_station['B']
            CMONOC_station_name = nearest_station['name'].lstrip().rstrip()
            radio_sound_lats.append(lat_radio_sound)  # 保存探空站经纬度和站名
            radio_sound_lons.append(lon_radio_sound)
            radio_sound_station_names.append(name_radio_sound)
            CMONOC_lats.append(CMONOC_lat)  # 保存陆态网经纬度和站名
            CMONOC_lons.append(CMONOC_lon)
            CMONOC_station_names.append(CMONOC_station_name)
            distances.append(nearest_station['distances'])
    target_dataframe = pd.DataFrame({
        'lon_radio_sound': radio_sound_lons,
        'lat_radio_sound': radio_sound_lats,
        'name_radio_sound': radio_sound_station_names,
        'lon_CMONOC': CMONOC_lons,
        'lat_CMONOC': CMONOC_lats,
        'name_CMONOC': CMONOC_station_names,
        'distance': distances
    })
    target_dataframe.to_excel('stations_col_CMONOC.xlsx')
