# -*- coding: utf-8 -*-
from matplotlib import pyplot as plt
import matplotlib as mlt
import pandas as pd
import numpy as np
import geopandas as gp


def plot_on_map(mis_list,obs_list, know_set, unknow_set, dataset='METR-LA', **kwargs):
    data_mis = np.array(mis_list)
    data_obs = np.array(obs_list)
    data = np.append(data_mis, data_obs)

    url_census = 'data/metr/Census_Road_2010_shapefile/Census_Road_2010.shp'
    meta_locations = pd.read_csv('data/metr/graph_sensor_locations.csv')
    map_metr = gp.read_file(url_census, encoding="utf-8")
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    map_metr.plot(ax=ax, color='black', alpha=0.2)
    lng_div = 0.01
    lat_div = 0.01
    ax.set_xlim((np.min(meta_locations['longitude']) - lng_div, np.max(meta_locations['longitude']) + lng_div))
    ax.set_ylim((np.min(meta_locations['latitude']) - lat_div, np.max(meta_locations['latitude']) + lat_div))
    ax.set_xticks([])
    ax.set_yticks([])
    # draw sensor positions
    known = ax.scatter(meta_locations['longitude'][list(know_set)], meta_locations['latitude'][list(know_set)],
                       s=20, cmap=plt.cm.cool, c=data_obs,
                       norm=mlt.colors.Normalize(vmin=data.min(), vmax=data.max()), alpha=1, label='Known nodes')
    unknown = ax.scatter(meta_locations['longitude'][list(unknow_set)], meta_locations['latitude'][list(unknow_set)],
                         s=40, cmap=plt.cm.cool, c=data_mis,
                         norm=mlt.colors.Normalize(vmin=data.min(), vmax=data.max()), alpha=1, marker='*',
                         label='Unknown nodes')

    l = 0.92
    b = 0.03
    w = 0.015
    h = 0.8
    rect = [l, b, w, h]
    cbar_ax = fig.add_axes(rect)
    cbar = fig.colorbar(known, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=15)
    plt.savefig('rmse.png', dpi=500)


"""
if __name__ == '__main__':
    plot_on_map()
"""
