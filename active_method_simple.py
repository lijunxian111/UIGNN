# -*- coding: utf-8 -*-
"""
Reproducing experimental results of LL4AL [Yoo et al. 2019 CVPR]
"""

from __future__ import division
import torch
import numpy as np
import torch.optim as optim
from torch import nn
import matplotlib.pyplot as plt
from utils import load_metr_la_rdata, get_normalized_adj, get_Laplace, calculate_random_walk_matrix,test_error,test_error_forecasting_active,load_nerl_data,load_sedata,load_udata,load_pems_data
import random
import pandas as pd
from basic_structure import D_GCN, C_GCN, K_GCN,IGNNK
from EDL_loss import EvidentialLossSumOfSquares
from vis import plot_on_map

import geopandas as gp
import matplotlib as mlt
from copy import deepcopy
import pickle as pkl

n_o_n_m = 150 #sampled space dimension
h = 24 #sampled time dimension
z = 100 #hidden dimension for graph convolution
K = 1 #If using diffusion convolution, the actual diffusion convolution step is K+1
n_m = 50 #number of mask node during training
N_u = 50 #target locations, N_u locations will be deleted from the training data
Max_episode = 150 #max training episode
learning_rate = 0.0001 #the learning_rate for Adam optimizer
E_maxvalue = 80 #the max value from experience
batch_size = 4
EDL=True
MC_dropout=False
dataset="metr_la"

"""
active learning settings
"""
TRIALS=3
CYCLES=18
ADDNUM=10

def enable_dropout(m):
    for each_module in m.modules():
        if each_module.__class__.__name__.startswith('Dropout'):
            each_module.train()

if __name__=="__main__":
    if dataset=="metr_la":
        A, X = load_metr_la_rdata()
    elif dataset == "nrel":
        A,X,files_info = load_nerl_data()
    elif dataset == 'sedata':
        A, X = load_sedata()
        A = A.astype('float32')
        X = X.astype('float32')
    elif dataset == 'pems':
        A,X = load_pems_data()
    elif dataset == 'ushcn':
        A,X,Omissing = load_udata()
        X = X[:,:,:,0]
        X = X.reshape(1218,120*12)
        X = X/100
    else:
        raise NotImplementedError('Please specify datasets from: metr, nrel, ushcn, sedata or pems')


    if dataset=="metr_la":
        split_line1 = int(X.shape[2] * 0.7)

        training_set = X[:, 0, :split_line1].transpose()

        test_set = X[:, 0, split_line1:].transpose()  # split the training and test period

        full_dataset = X[:, 0, :].transpose()
    elif dataset=="nrel":
        time_used_base = np.arange(84, 228)
        time_used = np.array([])
        for i in range(365):
            time_used = np.concatenate((time_used, time_used_base + 24 * 12 * i))
        X = X[:, time_used.astype(np.int)]
        capacities = np.array(files_info['capacity'])
        capacities = capacities.astype('float32')
        split_line1 = int(X.shape[1] * 0.7)
        training_set = X[:, :split_line1].transpose()
        test_set = X[:, split_line1:].transpose()  # split the training and test period
        full_dataset = X.transpose()
        E_maxvalue = capacities.max()
    elif dataset == "pems":
        X=X.transpose()
        split_line1 = int(X.shape[1] * 0.7)
        training_set = X[:, :split_line1].transpose()
        test_set = X[:, split_line1:].transpose()  # split the training and test period
        full_dataset = X.transpose()
    else:
        split_line1 = int(X.shape[1] * 0.7)
        training_set = X[:, :split_line1].transpose()
        test_set = X[:, split_line1:].transpose()  # split the training and test period
        full_dataset=X.transpose()

    """
    rand = np.random.RandomState(0)  # Fixed random output, just an example when seed = 0.
    unknow_set = rand.choice(list(range(0, X.shape[0])), N_u, replace=False)

    unknow_set = set(unknow_set)
    full_set = set(range(0, X.shape[0]))
    know_set = full_set - unknow_set
    training_set_s = training_set[:, list(know_set)]  # get the training data in the sample time period
    A_s = A[:, list(know_set)][list(know_set), :]  # get the observed adjacent matrix from the full adjacent matrix,
    # the adjacent matrix are based on pairwise distance,
    # so we need not to construct it for each batch, we just use index to find the dynamic adjacent matrix

    STmodel = IGNNK(h, z, K, forecasting=True,mc_dropout=MC_dropout,EDL=EDL)  # The graph neural networks

    if EDL:
        criterion=EvidentialLossSumOfSquares()
    else:
        criterion = nn.MSELoss()
    
    """

    for trial in range(TRIALS):
        MEAN_list = []
        STD_list = []

        RMSE_list = []
        MAE_list = []
        MAPE_list = []

        RMSE_mis_list = []
        MAE_mis_list = []
        MAPE_mis_list = []
        NLL_mis_list = []

        RMSE_obs_list = []
        MAE_obs_list = []
        MAPE_obs_list = []
        NLL_obs_list = []
        rand = np.random.RandomState(0)  # Fixed random output, just an example when seed = 0.
        know_set = rand.choice(list(range(0, X.shape[0])), 30, replace=False)
        full_set = set(range(0, X.shape[0]))
        know_set=  set(know_set)

        unknow_set=full_set-know_set  # get the training data in the sample time period
        A_s = A[:, list(know_set)][list(know_set), :]
         # get the observed adjacent matrix from the full adjacent matrix,
        # the adjacent matrix are based on pairwise distance,
        # so we need not to construct it for each batch, we just use index to find the dynamic adjacent matrix

        STmodel = IGNNK(h, z, K, forecasting=True, mc_dropout=MC_dropout, EDL=EDL)  # The graph neural networks

        for cycle in range(CYCLES-1):
            training_set_s = training_set[:, list(know_set)]
            if EDL:
                criterion = EvidentialLossSumOfSquares()
            else:
                criterion = nn.MSELoss()


            optimizer = optim.Adam(STmodel.parameters(), lr=learning_rate)

            STmodel.train()
            for epoch in range(Max_episode):

               STmodel.train()
               total_loss=0.
               cnt=0.
               for i in range(
                    training_set.shape[0] // (h * batch_size)):  # using time_length as reference to record test_error

               # forecasting for 15 mins
               #print(training_set_s.shape[0] - h - 3)
               #print(training_set_s.shape[1])
                   t_random = np.random.randint(0, high=(training_set_s.shape[0] - h - 6), size=batch_size, dtype='l')
                   know_mask = set(random.sample(range(0, training_set_s.shape[1]), 30+cycle*ADDNUM))  # sample n_o + n_m nodes
                   feed_batch = []
                   target_batch = []
                   for j in range(batch_size):
                        feed_batch.append(
                            training_set_s[t_random[j]: t_random[j] + h, :][:, list(know_mask)])  # generate 8 time batches
                        target_batch.append(training_set_s[t_random[j] + h + 5: t_random[j] + h + 6, :][:,
                                        list(know_mask)])  # generate target for 15 mins later forecasting
                #5，6 for 30mins,8,9 for 45mins

                   inputs = np.array(feed_batch)
                   inputs_omask = np.ones(np.shape(inputs))
                   inputs_omask[inputs == 0] = 0  # We found that there are irregular 0 values for METR-LA, so we treat those 0 values as missing data,
            # For other datasets, it is not necessary to mask 0 values
                   targets = np.array(target_batch)
                   targets_omask = np.ones(np.shape(targets))
                   targets_omask[targets == 0] = 0

                   missing_index = np.ones((inputs.shape))
            # missing_index_target = np.ones((targets.shape))
                   for j in range(batch_size):
                        missing_mask = random.sample(range(0, 30+cycle*ADDNUM), int((30+cycle*ADDNUM)/3))  # Masked locations
                        missing_index[j, :, missing_mask] = 0
                # missing_index_target[j, :, missing_mask] = 0

                   Mf_inputs = inputs * inputs_omask * missing_index / E_maxvalue  # normalize the value according to experience
            # Mf_targets = targets * targets_omask * missing_index_target / E_maxvalue

                   Mf_inputs = torch.from_numpy(Mf_inputs.astype('float32'))
            # Mf_targets = torch.from_numpy(Mf_targets.astype('float32'))

            # The errors on abnormal zeros are not used for training
                   mask = torch.from_numpy(targets_omask.astype('float32'))

                   A_dynamic = A_s[list(know_mask), :][:, list(know_mask)]  # Obtain the dynamic adjacent matrix
                   A_q = torch.from_numpy((calculate_random_walk_matrix(A_dynamic).T).astype('float32'))
                   A_h = torch.from_numpy((calculate_random_walk_matrix(A_dynamic.T).T).astype('float32'))

                   outputs = torch.from_numpy(targets / E_maxvalue)  # The label

                   optimizer.zero_grad()
                   if EDL:
                       X_res, out_prob=STmodel(Mf_inputs, A_q, A_h)
                       loss = criterion(out_prob * mask, outputs * mask)
                   else:
                       X_res,_ = STmodel(Mf_inputs, A_q, A_h)  # Obtain the reconstruction
                       loss = criterion(X_res * mask, outputs * mask)

                   total_loss+=loss.item()
                   cnt+=1
            #MEAN_list.append(X_mean)
            #STD_list.append(X_std)
                   loss.backward()
                   optimizer.step()  # Errors backward

               print(f"epoch{epoch}: train loss {total_loss/cnt}")

            #MEAN=torch.stack(MEAN_list,dim=0)
            #STD=torch.stack(STD_list,dim=0)
            #print(torch.mean(MEAN,dim=0).detach().squeeze().numpy())
            #print(torch.mean(STD, dim=0).detach().squeeze().numpy())

            STmodel.eval()
            if MC_dropout:
              STmodel.MC_dropout.train()



            obs, mis, overall = test_error_forecasting_active(STmodel, unknow_set, test_set, A, E_maxvalue, True, mc_drop=MC_dropout, EDL=EDL)
            RMSE_list.append(overall[0])
            MAE_list.append(overall[1])
            MAPE_list.append(overall[2])

            RMSE_mis_list.append(mis[0])
            MAE_mis_list.append(mis[1])
            MAPE_mis_list.append(mis[2])
            if EDL:
               NLL_mis_list.append(mis[4])

            RMSE_obs_list.append(obs[0])
            MAE_obs_list.append(obs[1])
            MAPE_obs_list.append(obs[2])
            if EDL:
               NLL_obs_list.append(obs[4])


            print(overall[0], overall[1], overall[2])
            print(f"mis_rmse_trial{trial}_cycle{cycle}",mis[1])
            print(f"obs_rmse_trial{trial}_cycle{cycle}",obs[1])
            if EDL:
                print(f"mis_nll_trial{trial}_cycle{cycle}", mis[4])
                print(f"obs_nll_trial{trial}_cycle{cycle}", obs[4])

            if MC_dropout or EDL:
                uncertainty=mis[3]
                arg = np.argsort(uncertainty)

                know_set=list(know_set)
                unknow_set=np.array(list(unknow_set))[arg]
                if(unknow_set.shape[0]<ADDNUM):
                    break
                know_set = set(know_set+unknow_set[-ADDNUM:].tolist())
                unknow_set = unknow_set[:-ADDNUM].tolist()

                training_set_s = training_set[:, list(know_set)]  # get the training data in the sample time period
                A_s = A[:, list(know_set)][list(know_set), :]

                print(f"TRIAL:{trial} CYCLE:{cycle} know_set length: {len(know_set)}")

            #plot_on_map(mis[3],obs[3],know_set,unknow_set)
        print("final rmse: ")
        print(RMSE_mis_list[-1].mean())
        print(RMSE_obs_list[-1].mean())
        if EDL:
            print("final nll: ")
            print(NLL_mis_list[-1].mean())
            print(NLL_obs_list[-1].mean())

        print("TOTAL:")
        print(RMSE_list)
        print("MIS:")
        print(RMSE_mis_list)
        print("OBS:")
        print(RMSE_obs_list)

        plt.figure()
        lenth=len(RMSE_obs_list)
        sample_num=np.array([i*10+50 for i in range(lenth)])
        plt.plot(sample_num,np.array(RMSE_obs_list),label='obs rmse')
        plt.plot(sample_num,np.array(RMSE_mis_list),label='mis rmse')
        plt.legend()
        plt.xlabel('sample num')
        plt.ylabel('RMSE')
        plt.title('RMSE with sample number changes')
        plt.savefig(f'TRIAL{trial+1}.png')
        plt.show()

        idx = np.argmin(np.array(RMSE_list))
        print(f'Best model result: all: {np.array(RMSE_list)[idx]} obs: {np.array(RMSE_obs_list)[idx]} mis: {np.array(RMSE_mis_list)[idx]}')

    """
    
    print("final rmse: ")
    print(RMSE_mis_list[-1].mean())
    print(RMSE_obs_list[-1].mean())
    if EDL:
        print("final nll: ")
        print(NLL_mis_list[-1].mean())
        print(NLL_obs_list[-1].mean())
    """

    idx = np.argmin(np.array(RMSE_list))
    print(f'Best model result: all: {np.array(RMSE_list)[idx]} obs: {np.array(RMSE_obs_list)[idx]} mis: {np.array(RMSE_mis_list)[idx]}')

