# -*- coding: utf-8 -*-
import numpy as np
import argparse
import baselines
from pathlib import Path
import os
import threading


parser = argparse.ArgumentParser()
parser.add_argument('--baseline', default='kriging', choices=['kpmf','knn','ignnk','gltl','kriging','stmlv'])
parser.add_argument('--dataset', default='pems', choices=['metr_la', 'pems','nrel','sedata','ushcn'])
args = parser.parse_args()


if __name__=='__main__':
    dataset = args.dataset
    baseline = args.baseline
    method = Path('run_' + baseline)
    prepareData = Path('baselines').joinpath(baseline).joinpath(method)

    os.system('python ' + str(prepareData) +  '.py ' + '--dataset ' + dataset)
    print(threading.get_ident())
    print('recovering data')

    # call function to process and recover data at folder
    # b1: not processed b2: processed


    # call forecasting function

    # report rmse for this method