import torch
import math

def masked_mae_loss(y_pred, y_true):
    mask = (y_true != 0).float()
    mask /= mask.mean()

    loss = torch.abs(y_pred - y_true)
    
    loss = loss * mask
    # trick for nans: https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/3
    loss[loss != loss] = 0
    return loss.mean()

def rmse_loss(y_pred, y_true):
    mask = (y_true != 0).float()
    mask /= mask.mean()

    y_pred = y_pred.mean(dim = 1)
    loss = (y_pred  - y_true) ** 2
    loss = loss * mask
    # trick for nans: https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/3
    loss[loss != loss] = 0
    return torch.sqrt(loss.mean())

def mae(y_pred, y_true):
    mask = (y_true != 0).float()
    mask /= mask.mean()

    y_pred_mean = y_pred.mean(-1)
    #y_pred_mean = y_pred_mean.unsqueeze(-1)

    #print("y_pred_mean.size = ",y_pred_mean.size())
    #print("y_true = ",y_true.size())

    loss = torch.abs(y_pred_mean - y_true)
    loss = loss * mask
    # trick for nans: https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/3
    loss[loss != loss] = 0
    return loss.mean()

def mis(y_pred, y_true):
    
    mask = (y_true != 0).float()
    mask /= mask.mean()

    mu = y_pred.mean(dim = 1)
    #mu = mu.unsqueeze(-1)

    sigam = torch.var(y_pred,dim=1)
    #mu = mu.unsqueeze(-1)

    rou = 0.05

    lx = mu - sigam.sqrt() * 1.96
    ux = mu + sigam.sqrt() * 1.96

    ans_mis = ux - lx 
    E = 0.025 * (mu - ux) + sigam.sqrt()/(math.sqrt((2*math.pi)))*torch.exp(-((ux-mu)**2)/(2*sigam))
    ans_mis = ans_mis + 4/rou * E

    ans_mis = ans_mis * mask
    # trick for nans: https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/3
    ans_mis[ans_mis != ans_mis] = 0
    return ans_mis.mean()

def interval_width(y_pred, y_true):
    
    mask = (y_true != 0).float()
    mask /= mask.mean()

    mu = y_pred.mean(dim = 1)
    #mu = mu.unsqueeze(-1)

    sigam = torch.var(y_pred,dim=1)

    rou = 0.05

    lx = mu - sigam.sqrt()*1.96
    ux = mu + sigam.sqrt()*1.96

    ans_width = ux - lx
    #ans_width[ans_width != ans_width] = 0
    ans_width = ans_width * mask
    # trick for nans: https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/3
    ans_width[ans_width != ans_width] = 0
    #losses = []
    #losses.append(ans_width.unsqueeze(0))
    #width = torch.mean(torch.sum(torch.cat(losses, dim=0), dim=0))
    return ans_width.mean()