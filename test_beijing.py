# -*- coding: utf-8 -*-
from __future__ import division
import torch
import numpy as np
import torch.optim as optim
from utils_full import load_beijing_data

if __name__ == "__main__":
    A, X, data = load_beijing_data()
    #print(A)
    #print(X)
    print(A.shape)
    print(X.shape)