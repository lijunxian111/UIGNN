# -*- coding: utf-8 -*-
import threading
from U_IGNNK.utils import load_metr_la_rdata
import argparse
import numpy as np
from U_IGNNK.basic_structure import IGNNK
import random
import torch
import torch.optim as optim
from U_IGNNK.utils import *
from copy import deepcopy

n_o_n_m = 240  # sampled space dimension
h = 24  # sampled time dimension
z = 100  # hidden dimension for graph convolution
K = 1  # If using diffusion convolution, the actual diffusion convolution step is K+1
n_m = 80  # number of mask node during training
N_u = 80  # target locations, N_u locations will be deleted from the training data
Max_episode = 200  # max training episode
learning_rate = 0.001  # the learning_rate for Adam optimizer
E_maxvalue = 80  # the max value from experience
batch_size = 4

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str,choices=['metr_la', 'pems','nrel','sedata','ushcn'],default='pems')
args = parser.parse_args()

if __name__ == "__main__":
    #device = 'cuda:0' if torch.has_cuda else 'cpu'
    device='cpu'
    device = torch.device(device)

    print(threading.get_ident())
    dataset = args.dataset
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
        A = A.astype('float32')
        X = X.astype('float32')
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


    rand = np.random.RandomState(0)  # Fixed random output, just an example when seed = 0.
    unknow_set = rand.choice(list(range(0, X.shape[0])), N_u, replace=False)

    unknow_set = set(unknow_set)
    full_set = set(range(0, X.shape[0]))
    know_set = full_set - unknow_set
    # neighbor is calculated from directional adjacent matrix
    # k = 5
    knn_training_extended = deepcopy(training_set)
    knn_training_extended[:, list(unknow_set)] = 0
    knn_training_original = training_set

    knn_test_extended = deepcopy(test_set)
    knn_test_extended[:, list(unknow_set)] = 0
    knn_test_original = test_set
    # assign 0 to non seen locations
    A_m = deepcopy(A)
    adj = A_m.T
    adj[:, list(unknow_set)] = 0
    adj -= np.eye(len(full_set))
    neighbor_idx = np.flip(np.argsort(adj, axis=1), axis=1)[:, :5]
    neighbor_training = np.mean(knn_training_extended[:, neighbor_idx], axis=-1)
    knn_training_extended[:, list(unknow_set)] = neighbor_training[:, list(unknow_set)]

    neighbor_test = np.mean(knn_test_extended[:, neighbor_idx], axis=-1)
    knn_test_extended[:, list(unknow_set)] = neighbor_test[:, list(unknow_set)]

    # forecasting model training
    FCmodel = IGNNK(h, z, K, forecasting=True)
    MEAN_list = []
    STD_list = []

    RMSE_list = []
    MAE_list = []
    MAPE_list = []

    RMSE_mis_list = []
    MAE_mis_list = []
    MAPE_mis_list = []

    RMSE_obs_list = []
    MAE_obs_list = []
    MAPE_obs_list = []
    FCmodel.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(FCmodel.parameters(), lr=learning_rate)

    for epoch in range(Max_episode):
       FCmodel.train()
       for i in range(
            training_set.shape[0] // (h * batch_size)):  # using time_length as reference to record test_error

        # forecasting for 15 mins
        t_random = np.random.randint(0, high=(knn_training_extended.shape[0] - h - 6), size=batch_size, dtype='l')
        feed_batch = []
        target_batch = []
        for j in range(batch_size):
            feed_batch.append(
                knn_training_extended[t_random[j]: t_random[j] + h, :])  # generate 8 time batches
            target_batch.append(knn_training_extended[t_random[j] + h + 5: t_random[j] + h + 6,
                                :])  # generate target for 15 mins later forecasting

        inputs = np.array(feed_batch)
        inputs_omask = np.ones(np.shape(inputs))
        inputs_omask[
            inputs == 0] = 0  # We found that there are irregular 0 values for METR-LA, so we treat those 0 values as missing data,
        # For other datasets, it is not necessary to mask 0 values
        targets = np.array(target_batch)
        targets_omask = np.ones(np.shape(targets))
        targets_omask[targets == 0] = 0

        # no missing mask in forecasting baseline model
        # missing_index = np.ones((inputs.shape))
        # # missing_index_target = np.ones((targets.shape))
        # for j in range(batch_size):
        #     missing_mask = random.sample(range(0, n_o_n_m), n_m)  # Masked locations
        #     missing_index[j, :, missing_mask] = 0
        #     # missing_index_target[j, :, missing_mask] = 0

        Mf_inputs = inputs * inputs_omask / E_maxvalue  # normalize the value according to experience
        # Mf_targets = targets * targets_omask * missing_index_target / E_maxvalue

        Mf_inputs = torch.from_numpy(Mf_inputs.astype('float32'))
        # Mf_targets = torch.from_numpy(Mf_targets.astype('float32'))

        # The errors on abnormal zeros are not used for training
        mask = torch.from_numpy(targets_omask.astype('float32'))

        A_q = torch.from_numpy((calculate_random_walk_matrix(A).T).astype('float32'))
        A_h = torch.from_numpy((calculate_random_walk_matrix(A.T).T).astype('float32'))

        outputs = torch.from_numpy(targets / E_maxvalue)  # The label

        optimizer.zero_grad()

        X_res, _ = FCmodel(Mf_inputs, A_q, A_h)  # Obtain the reconstruction
        loss = criterion(X_res * mask, outputs * mask)
        # MEAN_list.append(X_mean)
        # STD_list.append(X_std)

        loss.backward()
        optimizer.step()  # Errors backward
       FCmodel.eval()
       if epoch % 5 == 0:
        obs, mis, overall = test_error_forecasting_baseline(FCmodel, unknow_set, knn_test_extended, knn_test_original,
                                                            A, E_maxvalue, True)

        RMSE_list.append(overall[0])
        MAE_list.append(overall[1])
        MAPE_list.append(overall[2])

        RMSE_mis_list.append(mis[0])
        MAE_mis_list.append(mis[1])
        MAPE_mis_list.append(mis[2])

        RMSE_obs_list.append(obs[0])
        MAE_obs_list.append(obs[1])
        MAPE_obs_list.append(obs[2])
        print(f"Epoch{epoch}")

    #best_epoch = np.argmin(np.array(RMSE_list))

    best_epoch = np.argmin(np.array(RMSE_obs_list))
    print('MAE Best model result:', np.array(MAE_list)[best_epoch], np.array(MAE_obs_list)[best_epoch],
          np.array(MAE_mis_list)[best_epoch])
    print('RMSE Best model result:', np.array(RMSE_list)[best_epoch], np.array(RMSE_obs_list)[best_epoch],
          np.array(RMSE_mis_list)[best_epoch])
    print('MAPE Best model result:', np.array(MAPE_list)[best_epoch], np.array(MAPE_obs_list)[best_epoch],
          np.array(MAPE_mis_list)[best_epoch])
    #print('NLL Best model result:', np.array(NLL_obs_list)[best_epoch], np.array(NLL_mis_list)[best_epoch])
    print('FINISHED BASELINE RUNNING')