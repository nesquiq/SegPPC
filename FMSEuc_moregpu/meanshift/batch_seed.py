# Author Mengyang Zhao <Mengyang.Zhao@tufts.edu>

import math
import operator

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import exp, sqrt

def euc_batch(a, b):
    result = torch.cdist(a.unsqueeze(0),b.unsqueeze(0),p=2).squeeze()
    # result2 = sqrt(((b[None,:] - a[:,None]) ** 2).sum(2))
    # print(result)
    # print(result2)
    # print((result==result2))
    # print((result-result2).abs().sum())
    # torch.Size([128, 256]) torch.Size([50176, 256]) torch.Size([128, 50176, 256])
    # result = result.sum(2)
    # print(a.shape, b.shape, result.shape) # torch.Size([128, 50176])
    # result = sqrt(result)
    # print(a.shape, b.shape, result.shape)
    #pdist = torch.nn.PairwiseDistance(p=2)
    #result = pdist(a, b)
    return result

    #num = a@b.T
    #denom = torch.norm(a, dim=1).reshape(-1, 1) * torch.norm(b, dim=1)
    #return num / denom

def get_weight(sim, bandwidth):

    thr = 1-bandwidth
    #max = torch.tensor(1.0e+10).double().cuda()
    max = torch.tensor(1.0).float().cuda()
    min = torch.tensor(0.0).float().cuda()
    #dis=torch.where(sim>thr, 1-sim, max)
    dis=torch.where(sim>thr, max, min)

    return dis

def gaussian(dist, bandwidth):
    return exp(-0.5 * ((dist / bandwidth))**2) / (bandwidth * math.sqrt(2 * math.pi))

def meanshift_torch(X_gpu, seed, bandwidth, max_iter=300):

    stop_thresh = 1e-3 * bandwidth
    iter=0

    # X = torch.from_numpy(np.copy(data)).float().cuda()
    # S = torch.from_numpy(np.copy(seed)).float().cuda()
    # X = X_gpu.float()
    # S = seed.float()
    B = bandwidth.float().cuda()
    # B = torch.tensor(bandwidth).float().cuda()
    while True:
        #cosine = cos_batch(S, X)

        weight = gaussian(euc_batch(seed, X_gpu),B)

        #torch.where(distances>(1-bandwidth))
        #weight = gaussian(distances, B)
        num = (weight[:, :, None] * X_gpu).sum(dim=1)
        # S_old = S
        # S = num / weight.sum(1)[:, None]
        S_old = seed
        seed = num / weight.sum(1)[:, None]
        #cosine2 = torch.norm(S - S_old, dim=1).mean()
        iter+=1

        if (torch.norm(seed - S_old, dim=1).mean() < stop_thresh or iter == max_iter):
            break
    
    p_num=[]
    for line in weight:
        p_num.append(line[line==1].size()[0])

    # my_mean = seed.cpu().numpy()
    my_mean = seed

    return my_mean, p_num

