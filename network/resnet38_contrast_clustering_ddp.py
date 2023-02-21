import os
import sys
sys.path.append("/home/subin/Research/wseg_clustering")

import torch
import torch.nn as nn
import torch.sparse as sparse
import torch.nn.functional as F
import numpy as np
np.set_printoptions(threshold=np.inf)


import network.resnet38d
from tool import pyutils

"""
TODO: import extra modules
"""
import matplotlib.cm as cmap
from FMSEuc.meanshift.mean_shift_gpu import MeanShiftEuc
from sklearn import datasets, cluster
from sklearn.preprocessing import StandardScaler

def min_max_norm(inp):
    ret = (inp-inp.min())/(inp.max()-inp.min()+1e-7)
    return ret

def Clustering(f):
    # Normalize dataset for easier parameter selection
    with torch.no_grad():
        """
        cluster_methods2 reproduce
        """
        # f_cpu_flatten = f.detach().cpu().numpy().T
        # bw = 1.06*np.std(f_cpu_flatten)*(f_cpu_flatten.shape[0])**(-0.2)
        
        # ms = MeanShiftEuc(bandwidth=bw*0.3, GPU=True)
        # ms.fit(f_cpu_flatten)

        """
        author's recommendation
        """
        f_cpu = StandardScaler().fit_transform(f.detach().cpu().numpy()) # nhw, c
        f_max, f_min = f.max().detach().cpu().numpy(), f.min().detach().cpu().numpy()
        
        bandwidth = cluster.estimate_bandwidth(f_cpu[0:1499])
        bandwidth_gpu = bandwidth/(f_max-f_min)
        ms = MeanShiftEuc(bandwidth=bandwidth_gpu*0.15, cluster_all=True, GPU=True)
        # ms = MeanShiftEuc(bandwidth=bandwidth_gpu*0.05, cluster_all=True, GPU=True)
        ms.fit(f_cpu)
        """
        common
        """
        
        labels  =  ms.labels_
        centers = ms.cluster_centers_
    return labels, centers

def d2w_mapping(distances):
    weights = 1/(distances+1)
    return weights

def color_map(N, total):
    c_code = np.ones((1,1))*N
    c_code = cmap.jet(c_code/total)[:,:,:3]
    return c_code

def get_jet_img(img, cluster_result, hh, ww):
    out = cluster_result.reshape(1,1,hh,ww)/(cluster_result.max()+1e-7)
    out = F.interpolate(torch.from_numpy(out).float(), img.shape[-2:], mode='nearest').squeeze()
    jet = cmap.jet(out.cpu())[:,:,:3]
    out = (min_max_norm(img)*0.5+torch.from_numpy(jet).permute(2,0,1)).squeeze()
    out = out.permute(1,2,0).numpy()/(out.max()+1e-7)
    return out


class Net(network.resnet38d.Net):
    def __init__(self):
        super(Net, self).__init__()
        self.dropout7 = torch.nn.Dropout2d(0.5)
        self.fc8 = nn.Conv2d(4096, 21, 1, bias=False)
        self.fc_proj = torch.nn.Conv2d(4096, 128, 1, bias=False)
        self.fc_proj_256 = torch.nn.Conv2d(251, 256, 1, bias=False)
        # self.fc_proj_256 = torch.nn.Conv2d(259, 256, 1, bias=False)
        
        
        self.f8_0 = torch.nn.Conv2d(64, 8, 1, bias=False)
        self.f8_1 = torch.nn.Conv2d(128, 16, 1, bias=False)
        self.f8_2 = torch.nn.Conv2d(256, 32, 1, bias=False)
        self.f8_3 = torch.nn.Conv2d(512, 64, 1, bias=False)
        self.f8_4 = torch.nn.Conv2d(1024, 128, 1, bias=False)
        self.f9 = torch.nn.Conv2d(192+3, 192, 1, bias=False)

        """
        TODO: add branch to hook out intermediate feature from conv blocks
              --> done
        """

        torch.nn.init.xavier_uniform_(self.fc8.weight)
        torch.nn.init.kaiming_normal_(self.f8_0.weight)
        torch.nn.init.kaiming_normal_(self.f8_1.weight)
        torch.nn.init.kaiming_normal_(self.f8_2.weight)
        torch.nn.init.kaiming_normal_(self.f8_3.weight)
        torch.nn.init.kaiming_normal_(self.f8_4.weight)
        torch.nn.init.xavier_uniform_(self.f9.weight, gain=4)
        torch.nn.init.xavier_uniform_(self.fc_proj.weight)
        
        """
        TODO: set initialization for additional fc_XX layers
              for reducing channels of conv0, conv1, conv2, conv3
              --> done in init
        """

        self.from_scratch_layers = [self.f8_0, self.f8_1, self.f8_2, self.f8_3, self.f8_4, self.f9, self.fc8, self.fc_proj, self.fc_proj_256]
        
        """
        TODO: append additional fc_XX layers to from_scratch_layers
              --> done
        """

        self.not_training = [self.conv1a, self.b2, self.b2_1, self.b2_2]

    def forward(self, x):
        N, C, H, W = x.size()
        d = super().forward_as_dict(x)
        fea = self.dropout7(d['conv6'])

        f_proj = F.relu(self.fc_proj(fea), inplace=True)

        cam = self.fc8(fea)
        n,c,h,w = cam.size()
        x_s = F.interpolate(x, (h, w), mode='bilinear', align_corners=True)

        with torch.no_grad():
            cam_d = F.relu(cam.detach())
            cam_d_max = torch.max(cam_d.view(n, c, -1), dim=-1)[0].view(n, c, 1, 1)+1e-5
            # max norm
            cam_d_norm = F.relu(cam_d - 1e-5) / cam_d_max
            cam_d_norm[:, 0, :, :] = 1-torch.max(cam_d_norm[:, 1:, :, :], dim=1)[0]
            cam_max = torch.max(cam_d_norm[:,1:,:,:], dim=1, keepdim=True)[0]
            cam_d_norm[:,1:,:,:][cam_d_norm[:,1:,:,:] < cam_max] = 0
        
        f8_0 = F.relu(self.f8_0(d['conv1'].detach()), inplace=True)
        f8_1 = F.relu(self.f8_1(d['conv2'].detach()), inplace=True)
        f8_2 = F.relu(self.f8_2(d['conv3'].detach()), inplace=True)
        f8_3 = F.relu(self.f8_3(d['conv4'].detach()), inplace=True)
        f8_4 = F.relu(self.f8_4(d['conv5'].detach()), inplace=True)

        f_low_res = torch.cat([x_s, f8_3, f8_4], dim=1)
        n, c, h, w = f_low_res.size()

        cam_rv_down = self.PCM(cam_d_norm, f_low_res)
        cam_rv = F.interpolate(cam_rv_down, (H,W),
                            mode='bilinear', align_corners=True)

        n2, c2, h2, w2 = f8_2.shape
        x_s2 = F.interpolate(x, (h2, w2), mode='bilinear', align_corners=True)
        f8_0 = F.interpolate(f8_0, (h2, w2), mode='bilinear', align_corners=True)
        f8_1 = F.interpolate(f8_1, (h2, w2), mode='bilinear', align_corners=True)
        f8_2 = F.interpolate(f8_2, (h2, w2), mode='bilinear', align_corners=True)
        f8_3 = F.interpolate(f8_3, (h2, w2), mode='bilinear', align_corners=True)
        f8_4 = F.interpolate(f8_4, (h2, w2), mode='bilinear', align_corners=True)
        f_all_res = torch.cat([x_s2, f8_0, f8_1, f8_2, f8_3, f8_4], dim=1)

        f_all_res = self.fc_proj_256(f_all_res)
        cam = F.interpolate(cam, (H, W), mode='bilinear', align_corners=True)
        """
        TODO: conv0~conv3 feature --> f8_0~f8_3 layer
                --> done
            add concatenate conv0 to conv5
                --> done
            TODO in clustering module
            add projection of concatenated features
                --> done
            flatten_features --> num_pixels, channel
                --> done
            clustering --> n_clusters, channel
                --> done
            flatten_feature * cluster_result.T --> cluster_pixel_correlation_map --> n_pixels, n_prototypes
                --> done
            cluster_result from reduced channel_dimension --> align with original 4096 dim feature pixel
                --> done
            estimate prototype in 4096 dimesion
                --> done
            prototype_classification in 4096 dimension
                --> done
            final refined cam = cluster_pixel_correlation_map * prototype_classification
                --> done

            
            TODO in train.py
            adopt p2p_wseg's process for our refined cams and prototypes
            add losses
            
        """
        return cam, cam_rv, f_proj, cam_rv_down, f_all_res, None

    """
    TODO: define clustering functions with Fast_mean_shift_EUC
    """

    def PCM(self, cam, f):

        n,c,h,w = f.size()
        cam = F.interpolate(cam, (h,w), mode='bilinear', align_corners=True).view(n,-1,h*w)
        f = self.f9(f)
        f = f.view(n, -1, h*w)
        # norm
        f = f / (torch.norm(f, dim=1, keepdim=True) + 1e-5)
        aff = F.relu(torch.matmul(f.transpose(1, 2), f), inplace=True)
        aff = aff/(torch.sum(aff, dim=1, keepdim=True) + 1e-5)
        cam_rv = torch.matmul(cam, aff).view(n, -1, h, w)

        return cam_rv

    def get_parameter_groups(self):
        groups = ([], [], [], [])
        print('======================================================')
        for m in self.modules():

            if (isinstance(m, nn.Conv2d) or isinstance(m, nn.modules.normalization.GroupNorm)):

                if m.weight.requires_grad:
                    if m in self.from_scratch_layers:
                        groups[2].append(m.weight)
                    else:
                        groups[0].append(m.weight)

                if m.bias is not None and m.bias.requires_grad:
                    if m in self.from_scratch_layers:
                        groups[3].append(m.bias)
                    else:
                        groups[1].append(m.bias)

        return groups

class Net_backup(network.resnet38d.Net):
    def __init__(self):
        super(Net, self).__init__()
        self.dropout7 = torch.nn.Dropout2d(0.5)
        self.fc8 = nn.Conv2d(4096, 21, 1, bias=False)
        self.fc_proj = torch.nn.Conv2d(4096, 128, 1, bias=False)
        
        self.f8_3 = torch.nn.Conv2d(512, 64, 1, bias=False)
        self.f8_4 = torch.nn.Conv2d(1024, 128, 1, bias=False)
        self.f9 = torch.nn.Conv2d(192+3, 192, 1, bias=False)

        """
        TODO: add branch to hook out intermediate feature from conv blocks
        """

        torch.nn.init.xavier_uniform_(self.fc8.weight)
        torch.nn.init.kaiming_normal_(self.f8_3.weight)
        torch.nn.init.kaiming_normal_(self.f8_4.weight)
        torch.nn.init.xavier_uniform_(self.f9.weight, gain=4)
        torch.nn.init.xavier_uniform_(self.fc_proj.weight)
        
        """
        TODO: set initialization for additional fc_XX layers
              for reducing channels of conv0, conv1, conv2, conv3
        """

        self.from_scratch_layers = [self.f8_3, self.f8_4, self.f9, self.fc8, self.fc_proj]
        
        """
        TODO: append additional fc_XX layers to from_scratch_layers
        """

        self.not_training = [self.conv1a, self.b2, self.b2_1, self.b2_2]

    def forward(self, x):
        N, C, H, W = x.size()
        d = super().forward_as_dict(x)
        fea = self.dropout7(d['conv6'])

        f_proj = F.relu(self.fc_proj(fea), inplace=True)

        cam = self.fc8(fea)
        n,c,h,w = cam.size()

        with torch.no_grad():
            cam_d = F.relu(cam.detach())
            cam_d_max = torch.max(cam_d.view(n, c, -1), dim=-1)[0].view(n, c, 1, 1)+1e-5
            # max norm
            cam_d_norm = F.relu(cam_d - 1e-5) / cam_d_max
            cam_d_norm[:, 0, :, :] = 1-torch.max(cam_d_norm[:, 1:, :, :], dim=1)[0]
            cam_max = torch.max(cam_d_norm[:,1:,:,:], dim=1, keepdim=True)[0]
            cam_d_norm[:,1:,:,:][cam_d_norm[:,1:,:,:] < cam_max] = 0

        f8_3 = F.relu(self.f8_3(d['conv4'].detach()), inplace=True)
        f8_4 = F.relu(self.f8_4(d['conv5'].detach()), inplace=True)
        x_s = F.interpolate(x, (h, w), mode='bilinear', align_corners=True)
        f = torch.cat([x_s, f8_3, f8_4], dim=1)
        n, c, h, w = f.size()

        cam_rv_down = self.PCM(cam_d_norm, f)
        cam_rv = F.interpolate(cam_rv_down, (H,W),
                               mode='bilinear', align_corners=True)
        cam = F.interpolate(cam, (H, W), mode='bilinear', align_corners=True)
        
        """
        TODO: add concatenate conv0 to conv6
              add projection of concatenated features
              flatten_features --> num_pixels, channel
              clustering --> n_clusters, channel
              flatten_feature * cluster_result.T --> cluster_pixel_correlation_map --> n_pixels, n_prototypes
              cluster_result from reduced channel_dimension --> aling with original 4096 dim feature pixel
              estimate prototype in 4096 dimesion
              prototype_classification in 4096 dimension
              final refined cam = cluster_pixel_correlation_map * prototype_classification

              adopt p2p_wseg's process for our refined cams and prototypes

              add losses
              
        """



        return cam, cam_rv, f_proj, cam_rv_down