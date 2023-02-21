# python contrast_clustering_train_v02_repro_gpu3_v2.py --session_name subin_v02_repro_gpu3_b4_16ep_v2_100_10_30_bwde64_ranseed_svr7_221108 --network network.resnet38_contrast_clustering_repro --lr 0.01 --num_workers 8 --train_list voc12/train.txt --weights /home/subin/Research/hp_tuning_moreGPU/pretrained/ilsvrc-cls_rna-a1_cls1000_ep-0001.params --voc12_root /home/subin/Datasets/VOC2012/VOCdevkit/VOC2012 --tblog_dir ./tblog_reproduce_v02 --batch_size 4 --max_epoches 16

## BW normalization 수정
## BW adj double bounded

import os
import numpy as np
import torch
import random
import cv2

from torch.utils.data import DataLoader
from torchvision import transforms
import voc12.data
from tool import pyutils, imutils, torchutils, visualization
import argparse
import importlib
import torch.nn.functional as F
 
from FMSEuc_moregpu.meanshift.mean_shift_gpu import MeanShiftEuc
from sklearn import datasets, cluster
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

def KNN(X, target, k):
    dist_mat = torch.cdist(X, target)
    dist, idx = dist_mat.topk(k=k, dim=0, largest=False)
    return dist, idx


def adaptive_min_pooling_loss(x):
    # This loss does not affect the highest performance,
    # but change the optimial background score (alpha)
    n, c, h, w = x.size()
    k = h * w // 4
    x = torch.max(x, dim=1)[0]
    y = torch.topk(x.view(n, -1), k=k, dim=-1, largest=False)[0]
    y = F.relu(y, inplace=False)
    loss = torch.sum(y) / (k * n)

    return loss


def max_onehot(x):
    n, c, h, w = x.size()
    x_max = torch.max(x[:, 1:, :, :], dim=1, keepdim=True)[0]
    x[:, 1:, :, :][x[:, 1:, :, :] != x_max] = 0

    return x


def cls_proto_gen(f_proj, cam_rv_down, label, bg_th):
    with torch.no_grad():
        fea = f_proj.detach()
        c_fea = fea.shape[1]
        cam_rv_down = F.relu(cam_rv_down.detach())
        # ~(0,1) norm
        n, c, h, w = cam_rv_down.shape
        max = torch.max(cam_rv_down.view(n, c, -1), dim=-1)[0].view(n, c, 1, 1)
        min = torch.min(cam_rv_down.view(n, c, -1), dim=-1)[0].view(n, c, 1, 1)
        cam_rv_down[cam_rv_down < min + 1e-5] = 0.
        norm_cam = (cam_rv_down - min - 1e-5) / (max - min + 1e-5)
        cam_rv_down = norm_cam
        cam_rv_down[:, 0, :, :] = bg_th
        scores = F.softmax(cam_rv_down * label, dim=1)

        pseudo_label = scores.argmax(dim=1, keepdim=True)
        n_sc, c_sc, h_sc, w_sc = scores.shape
        fea = fea.permute(0, 2, 3, 1).reshape(-1, c_fea)

        top_values, top_indices = torch.topk(cam_rv_down.transpose(0, 1).reshape(c_sc, -1),
                                            k=h_sc * w_sc // 8, dim=-1)

        prototypes = torch.zeros(c_sc, c_fea).cuda()
        for i in range(c_sc):
            top_fea = fea[top_indices[i]]
            prototypes[i] = torch.sum(top_values[i].unsqueeze(-1) * top_fea, dim=0) / torch.sum(top_values[i])
        prototypes = F.normalize(prototypes, dim=-1)

    return prototypes, pseudo_label

def gen_batches(n, batch_size, *, min_batch_size=0):
    start = 0
    for _ in range(int(n // batch_size)):
        end = start + batch_size
        if end + min_batch_size > n:
            continue
        yield slice(start, end)
        start = end
    if start < n:
        yield slice(start, n)



def estimate_bw(X):
    n_neighbors = int(X.shape[0] * 0.3)

    if n_neighbors < 1:  
        n_neighbors = 1

    bandwidth = 0.0

    for batch in gen_batches(len(X), 500):
        d, _ = KNN(X, X[batch, :], n_neighbors)
        bandwidth += torch.max(d.T, axis=1)[0].sum()
    return bandwidth / X.shape[0]

# def estimate_bandwidth(X, quantile=0.3, n_samples=None, random_state=0, n_jobs=None):
#     ## is np ndarray? is nan in array?
#     ## n_neighbors
#     n_neighbors = int(X.shape[0] * quantile)

#     if n_neighbors < 1:  
#         n_neighbors = 1

#     nbrs = NearestNeighbors(n_neighbors=n_neighbors,
#                             n_jobs=n_jobs)
#     nbrs.fit(X)

#     bandwidth = 0.

#     for batch in gen_batches(len(X), 500):
#         d, _ = nbrs.kneighbors(X[batch, :], return_distance=True)
#         bandwidth += np.max(d, axis=1).sum()

#     return bandwidth / X.shape[0]

def clustering(f):
    # Normalize dataset for easier parameter selection
    with torch.no_grad():

        f = f.detach()
        m = f.mean(0, keepdim=True)
        s = f.std(0, unbiased=False, keepdim=True)
        
        f = f-m
        f = f/(s+1e-5)
        
        f_max, f_min = f.max().item(), f.min().item()

        # f_cpu = f.cpu().numpy()
        
        num_sample = 1500
        sample_id = np.arange(num_sample) * (f.shape[0]/num_sample)
        sample_id = sample_id.astype('int')
        # bandwidth_cpu = estimate_bandwidth(f_cpu[sample_id])
        # bandwidth_cpu = cluster.estimate_bandwidth(f_cpu[sample_id])
        bandwidth = estimate_bw(f[sample_id])
        # print(bandwidth,bandwidth_cpu)
        bandwidth_gpu = bandwidth/(f_max-f_min+1e-5)

        ms = MeanShiftEuc(bandwidth=bandwidth_gpu*0.25, cluster_all=True, GPU=True)
        ms.fit(f)

        # f_cpu = StandardScaler().fit_transform(f.detach().cpu().numpy()) # nhw, c
        # f_max, f_min = f.max().detach().cpu().numpy(), f.min().detach().cpu().numpy()
        
        # bandwidth = cluster.estimate_bandwidth(f_cpu[0:1499])
        # bandwidth_gpu = bandwidth/(f_max-f_min)
        # ms = MeanShiftEuc(bandwidth=bandwidth_gpu*0.01, cluster_all=True, GPU=True)
        # # ms = MeanShiftEuc(bandwidth=bandwidth_gpu*0.05, cluster_all=True, GPU=True)
        # ms.fit(f_cpu)
        """
        common
        """        
        labels  = ms.labels_
        centers = ms.cluster_centers_

    return labels, centers


def local_prototype_generation(f_all_res):
    n, c, h, w = f_all_res.size()
    f_all_res = f_all_res.detach().permute(1,0,2,3) # c, n, h, w
    f_all_res = f_all_res.reshape(c, n*h*w) # c, nhw
    f_all_res = f_all_res.permute(1,0) # nhw, c

    cluster_labels, cluster_centers = clustering(f_all_res)
    # cluster_labels = torch.from_numpy(cluster_labels)
    # cluster_centers = torch.from_numpy(cluster_centers)

    prototype_256 = torch.zeros((len(cluster_centers)),256)
    
    for i, c in enumerate(torch.unique(cluster_labels)):
        prototype_256[i] = f_all_res[torch.where(cluster_labels==c)[0]].mean()
    """
    TODO: prototype_256 쓸 것인가 아니면 cluster_centers 쓸 것인가?
    """
    return cluster_labels, cluster_centers


def prototype_association(proto1, proto2, labels1, labels2):
    sim_local_proto = torch.mm(proto1, proto2.T)
    
    proto_asso1 = torch.argmax(sim_local_proto, dim=1)
    proto_asso2 = torch.argmax(sim_local_proto, dim=0)

    asso_labels1 = proto_asso1[labels1]
    asso_labels2 = proto_asso2[labels2]

    return asso_labels1, asso_labels2

def prototype_association_iou(proto1, proto2, labels1, labels2):
    cluster1_one_hot = F.one_hot(labels1, num_classes = len(proto1)).float() ## nhw, num_cluster1
    cluster2_one_hot = F.one_hot(labels2, num_classes = len(proto2)).float() ## nhw, num_cluster2

    cluster_ovlap_intersection = cluster1_one_hot.T @ cluster2_one_hot ## num_cluster1, num_cluster2
    
    cluster_ovlap_union = (cluster1_one_hot.unsqueeze(-1)+cluster2_one_hot.unsqueeze(1)).sum(0)-cluster_ovlap_intersection
    
    cluster_iou = cluster_ovlap_intersection/(cluster_ovlap_union+1e-7)

    proto_asso1 = torch.argmax(cluster_iou, dim=1) ## num_cluster1 --> 0~ num_cluster2
    proto_asso2 = torch.argmax(cluster_iou, dim=0) ## num_cluster2 --> 0~ num_cluster1

    asso_labels1 = proto_asso1[labels1]
    asso_labels2 = proto_asso2[labels2]

    # ## cross corr ##
    # f1_proto_corr_map = f1_all_res@local_prototype2_256[proto_association1].cuda().T ## nhw, c * c, num_cluster1
    # f2_proto_corr_map = f2_all_res@local_prototype1_256[proto_association2].cuda().T ## nhw, c * c, num_cluster2

    return asso_labels1, asso_labels2
    

def loss_cross(f1, f2, pos1, neg1, pos2, neg2):
    # 1.1 cross-prototype      
    A1 = torch.exp(torch.sum(f1 * pos1, dim=-1) / 0.1)
    A2 = torch.sum(torch.exp(torch.matmul(f1, neg1.transpose(0, 1)) / 0.1), dim=-1)
    loss_nce1 = torch.mean(-1 * torch.log(A1 / A2))

    A3 = torch.exp(torch.sum(f2 * pos2, dim=-1) / 0.1)
    A4 = torch.sum(torch.exp(torch.matmul(f2, neg2.transpose(0, 1)) / 0.1), dim=-1)
    loss_nce2 = torch.mean(-1 * torch.log(A3 / A4))

    loss_cross_nce1 = 0.1 * (loss_nce1 + loss_nce2) / 2

    # 1.2 cross-pseudo-label
    A1_view1 = torch.exp(torch.sum(f1 * pos2, dim=-1) / 0.1)
    A2_view1 = torch.sum(torch.exp(torch.matmul(f1, neg2.transpose(0, 1)) / 0.1), dim=-1)
    loss_cross_nce2_1 = torch.mean(-1 * torch.log(A1_view1 / A2_view1))

    A3_view2 = torch.exp(torch.sum(f2 * pos1, dim=-1) / 0.1)
    A4_view2 = torch.sum(torch.exp(torch.matmul(f2, neg1.transpose(0, 1)) / 0.1), dim=-1)

    loss_cross_nce2_2 = torch.mean(-1 * torch.log(A3_view2 / A4_view2))

    loss_cross_nce2 = 0.1 * (loss_cross_nce2_1 + loss_cross_nce2_2) / 2

    return loss_cross_nce1, loss_cross_nce2


def loss_intra(f, pos, neg, label, num_proto, f_orig_size):
    n,_,h,w  = f_orig_size

    # semi-hard prototype mining
    similarity_intra = (torch.sum(f * pos, dim=-1) + 1) / 2.
    A1_intra_view = torch.exp(torch.sum(f * pos, dim=-1) / 0.1)
    neg_score = torch.matmul(f, neg.transpose(0, 1))
    with torch.no_grad():  # random 50%
        random_indices = torch.tensor([random.sample(range(num_proto), num_proto//2) for _ in range(n * h * w)]).long()
    with torch.no_grad():
        _, lower_indices = torch.topk(neg_score, k=neg_score.shape[-1]//2+1, largest=True, dim=-1)
        lower_indices = lower_indices[:, 3:]
    neg = neg.unsqueeze(0).repeat(n * h * w, 1, 1)
    # rand_neg = neg[torch.arange(n * h * w).unsqueeze(1), random_indices]
    lowr_neg = neg[torch.arange(n * h * w).unsqueeze(1), lower_indices]
    neg = torch.cat([pos.unsqueeze(1), lowr_neg], dim=1)
    A2_intra_view = torch.sum(torch.exp(torch.matmul(f.unsqueeze(1), neg.transpose(1, 2)).squeeze(1) / 0.1), dim=-1)
    loss_intra_nce = torch.zeros(1).cuda()
    C = 0
    local_exists = np.unique(label.cpu().numpy()).tolist()
    # hard pixel sampling
    for i_ in range(num_proto):  # for each class
        if not i_ in local_exists:
            continue
        C += 1
        A1_intra_view_class = A1_intra_view[label == i_]
        A2_intra_view_class = A2_intra_view[label == i_]
        similarity_intra_class = similarity_intra[label == i_]
        len_class = A1_intra_view_class.shape[0]
        if len_class < 2:
            continue

        with torch.no_grad():  # random 50%
            random_indices = torch.tensor(random.sample(range(len_class), len_class // 2)).long()
        random_A1_intra_view = A1_intra_view_class[random_indices]  # (n, hw//2)
        random_A2_intra_view = A2_intra_view_class[random_indices]

        with torch.no_grad():
            _, lower_indices = torch.topk(similarity_intra_class, k=int(len_class * 0.6), largest=False)
            lower_indices = lower_indices[int(len_class * 0.6) - len_class // 2:]

        lower_A1_intra_view = A1_intra_view_class[lower_indices]
        lower_A2_intra_view = A2_intra_view_class[lower_indices]

        A1_intra_view_class = torch.cat([random_A1_intra_view, lower_A1_intra_view], dim=0)  # (hw)
        A2_intra_view_class = torch.cat([random_A2_intra_view, lower_A2_intra_view], dim=0)
        A1_intra_view_class = A1_intra_view_class.reshape(-1)
        A2_intra_view_class = A2_intra_view_class.reshape(-1)
        loss_intra_nce += torch.mean(-1 * torch.log(A1_intra_view_class / A2_intra_view_class))

    # mean over existing prototyope
    loss_intra_nce = loss_intra_nce / C
    return loss_intra_nce


def loss_local(f1, f2):
    with torch.no_grad():
        f1_orig_size = f1.size()
        f2 = F.interpolate(f2, f1_orig_size[2:], mode='bilinear')

        clust_labels1, local_proto1_256 = local_prototype_generation(f1)
        clust_labels2, local_proto2_256 = local_prototype_generation(f2)
        
        local_proto1_256 = F.normalize(local_proto1_256, dim=-1)
        local_proto2_256 = F.normalize(local_proto2_256, dim=-1)

        # clust_asso_labels1, clust_asso_labels2 = prototype_association(local_proto1_256, local_proto2_256, clust_labels1, clust_labels2)
        clust_asso_labels1, clust_asso_labels2 = prototype_association_iou(local_proto1_256, local_proto2_256, clust_labels1, clust_labels2)
    
    # 1. cross-view contrastive learning
    local_pos1 = local_proto2_256[clust_asso_labels1].cuda()
    local_neg1 = local_proto2_256.cuda()

    local_pos2 = local_proto1_256[clust_asso_labels2].cuda()
    local_neg2 = local_proto1_256.cuda()

    ch = f1.size()[1]
    f1 = f1.permute(0,2,3,1).reshape(-1,ch)
    f1 = F.normalize(f1, dim=-1)
    f2 = f2.permute(0,2,3,1).reshape(-1,ch)
    f2 = F.normalize(f2, dim=-1)

    loss_local_cross_nce1, loss_local_cross_nce2 = loss_cross(f1, f2, local_pos1, local_neg1, local_pos2, local_neg2)
    
    # 2. intra-view contrastive learning
    # for source
    num_cluster1 = len(local_proto1_256)
    local_pos_intra1 = local_proto1_256[clust_labels1].cuda()
    local_neg_intra1 = local_proto1_256.cuda()
    
    # for target
    num_cluster2 = len(local_proto2_256)
    local_pos_intra2 = local_proto2_256[clust_labels2].cuda()
    local_neg_intra2 = local_proto2_256.cuda()

    loss_local_intra1 = loss_intra(f1, local_pos_intra1, local_neg_intra1, clust_labels1, num_cluster1, f1_orig_size)
    loss_local_intra2 = loss_intra(f2, local_pos_intra2, local_neg_intra2, clust_labels2, num_cluster2, f1_orig_size)
    loss_local_intra_nce = 0.1 * (loss_local_intra1 + loss_local_intra2) / 2

    return loss_local_cross_nce1, loss_local_cross_nce2, loss_local_intra_nce




# def loss_local_intra(f1, f2, local_proto1_256, local_proto2_256, clust_labels1, clust_labels2, f_orig_size):
#     # 2. intra-view contrastive learning
#     # for source
#     num_cluster1 = len(local_proto1_256)
#     local_pos_intra1 = local_proto1_256[clust_labels1].cuda()
#     local_neg_intra1 = local_proto1_256.cuda()

#     # for target
#     num_cluster2 = len(local_proto2_256)
#     local_pos_intra2 = local_proto2_256[clust_labels2].cuda()
#     local_neg_intra2 = local_proto2_256.cuda()

#     loss_local_intra1 = loss_intra(f1, local_pos_intra1, local_neg_intra1, clust_labels1, num_cluster1, f_orig_size)

#     loss_local_intra2 = loss_intra(f2, local_pos_intra2, local_neg_intra2, clust_labels2, num_cluster2, f_orig_size)
    
#     loss_local_intra_nce = 0.1 * (loss_local_intra1 + loss_local_intra2) / 2
    
#     return loss_local_intra_nce

