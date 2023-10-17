# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

class quantile_loss(nn.Module):
    def __init__(self):
        super(quantile_loss,self).__init__()
        self.quantiles = [0.05, 0.5, 0.95]
    def forward(self, pred, label):
        losses = []
        for i, q in enumerate(self.quantiles):
            diff = pred - label
            losses.append(torch.mean(torch.maximum((q - 1) * diff, q * diff)).view(1,1))
        loss = torch.mean(torch.cat(losses, dim=0))
        #print(loss)
        return loss