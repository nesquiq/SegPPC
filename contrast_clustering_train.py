# python contrast_clustering_train_v02_based_modulize_1116.py --session_name test_8batch --network network.resnet38_contrast_clustering --lr 0.01 --num_workers 8 --train_list voc12/train_aug.txt --weights /home/subin/Research/hp_tuning_moreGPU/pretrained/ilsvrc-cls_rna-a1_cls1000_ep-0001.params --voc12_root /home/subin/Datasets/VOC2012/VOCdevkit/VOC2012 --tblog_dir ./tblog_reproduce_v02 --batch_size 8 --max_epoches 8
## BW normalization 수정
## BW adj double bounded

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

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
from tensorboardX import SummaryWriter
import torch.nn.functional as F

"""
TODO: import extra modules
"""
# import wandb
from time import time
"""

t1 = time()

t2 = time()
elapsed = t2 - t1

wandb.log({'XXX_time': elapsed})
"""
from train_utils import cls_proto_gen, loss_intra, loss_local, loss_cross, adaptive_min_pooling_loss, max_onehot


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--max_epoches", default=8, type=int)
    # parser.add_argument("--network", default="network.resnet38_contrast", type=str)
    parser.add_argument("--network", default="network.resnet38_contrast_clustering", type=str)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--wt_dec", default=5e-4, type=float)
    parser.add_argument("--train_list", default="voc12/train_aug.txt", type=str)
    parser.add_argument("--val_list", default="voc12/val.txt", type=str)
    parser.add_argument("--session_name", default="resnet38_contrast", type=str)
    parser.add_argument("--crop_size", default=448, type=int)
    parser.add_argument("--weights", required=True, type=str)
    parser.add_argument("--voc12_root", default='VOC2012', type=str)
    parser.add_argument("--tblog_dir", default='./tblog', type=str)
    parser.add_argument("--bg_threshold", default=0.20, type=float)
    # parser.add_argument("--saved_dir", default='VOC2012', type=str)

    args = parser.parse_args()
    
    ## wandb
    # wandb.init(project="Your_Project_Name_Here", entity="Your_entity_here")
    # wandb.run.name = args.session_name
    # wandb.config.update(args)
    ##

    pyutils.Logger(args.session_name + '.log')

    print(vars(args))

    model = getattr(importlib.import_module(args.network), 'Net')()

    tblogger = SummaryWriter(args.tblog_dir)

    train_dataset = voc12.data.VOC12ClsDataset(args.train_list, voc12_root=args.voc12_root,
                                               transform=transforms.Compose([
                                                   imutils.RandomResizeLong(448, 768),
                                                   transforms.RandomHorizontalFlip(),
                                                   transforms.ColorJitter(brightness=0.3, contrast=0.3,
                                                                          saturation=0.3, hue=0.1),
                                                   np.asarray,
                                                   model.normalize,
                                                   imutils.RandomCrop(args.crop_size),
                                                   imutils.HWC_to_CHW,
                                                   torch.from_numpy
                                               ]))

    def worker_init_fn(worker_id):
        np.random.seed(1 + worker_id)

    train_data_loader = DataLoader(train_dataset,
                                   batch_size=args.batch_size,
                                   shuffle=True,
                                   num_workers=args.num_workers,
                                   pin_memory=True,
                                   drop_last=True,
                                   worker_init_fn=worker_init_fn)

    max_step = len(train_dataset) // args.batch_size * args.max_epoches

    param_groups = model.get_parameter_groups()
    optimizer = torchutils.PolyOptimizer([
        {'params': param_groups[0], 'lr': args.lr, 'weight_decay': args.wt_dec},
        {'params': param_groups[1], 'lr': 2 * args.lr, 'weight_decay': 0},
        {'params': param_groups[2], 'lr': 10 * args.lr, 'weight_decay': args.wt_dec},
        {'params': param_groups[3], 'lr': 20 * args.lr, 'weight_decay': 0}
    ], lr=args.lr, weight_decay=args.wt_dec, max_step=max_step)

    if args.weights[-7:] == '.params':
        import network.resnet38d

        assert 'resnet38' in args.network
        weights_dict = network.resnet38d.convert_mxnet_to_torch(args.weights)
    else:
        weights_dict = torch.load(args.weights)

    model.load_state_dict(weights_dict, strict=False)

    model = torch.nn.DataParallel(model).cuda()
    model.train()

    avg_meter = pyutils.AverageMeter('loss',
                                     'loss_cls',
                                     'loss_er',
                                     'loss_ecr',
                                     'loss_nce',
                                     'loss_intra_nce',
                                     'loss_cross_nce',
                                     'loss_cross_nce2',
                                     'loss_local_nce',
                                     'loss_local_intra_nce',
                                     'loss_local_cross_nce',
                                     'loss_local_cross_nce2')

    timer = pyutils.Timer("Session started: ")

    # Prototype
    PROTO1 = F.normalize(torch.rand(21, 128).cuda(), p=2, dim=1)
    PROTO2 = F.normalize(torch.rand(21, 128).cuda(), p=2, dim=1)
    for ep in range(args.max_epoches):
        
        for iter, pack in enumerate(train_data_loader):
            # scale_factor = 0.3
            ##
            t1 = time()
            ##
            img1 = pack[1]
            img2 = F.interpolate(img1,
                                 size=(128, 128),
                                 mode='bilinear',
                                 align_corners=True)
            N, C, H, W = img1.size()
            label = pack[2]

            ##
            t2 = time()
            time_pre_process = t1 -t2
            ##

            bg_score = torch.ones((N, 1))
            label = torch.cat((bg_score, label), dim=1)
            label = label.cuda(non_blocking=True).unsqueeze(2).unsqueeze(3)
            # cam1, cam_rv1, f_proj1, cam_rv1_down = model(img1)
            cam1, cam_rv1, f_proj1, cam_rv1_down, f1_all_res, _ = model(img1)
            ##
            t1 = time()
            time_forward1 = t2-t1
            ##
            label1 = F.adaptive_avg_pool2d(cam1, (1, 1))
            loss_rvmin1 = adaptive_min_pooling_loss((cam_rv1 * label)[:, 1:, :, :])

            cam1 = F.interpolate(visualization.max_norm(cam1),
                                 size=(128, 128),
                                 mode='bilinear',
                                 align_corners=True) * label
            cam_rv1 = F.interpolate(visualization.max_norm(cam_rv1),
                                    size=(128, 128),
                                    mode='bilinear',
                                    align_corners=True) * label
            ##
            t2 = time()
            time_post_process1 = t1 -t2
            ##

            # cam2, cam_rv2, f_proj2, cam_rv2_down = model(img2)
            cam2, cam_rv2, f_proj2, cam_rv2_down, f2_all_res, _ = model(img2)
            ##
            t1 = time()
            time_forward2 = t2-t1
            ##
            label2 = F.adaptive_avg_pool2d(cam2, (1, 1))
            loss_rvmin2 = adaptive_min_pooling_loss((cam_rv2 * label)[:, 1:, :, :])
            cam2 = visualization.max_norm(cam2) * label
            cam_rv2 = visualization.max_norm(cam_rv2) * label
            ##
            t2 = time()
            time_post_process2 = t1 -t2
            ##
            loss_cls1 = F.multilabel_soft_margin_loss(label1[:, 1:, :, :], label[:, 1:, :, :])
            loss_cls2 = F.multilabel_soft_margin_loss(label2[:, 1:, :, :], label[:, 1:, :, :])

            ns, cs, hs, ws = cam2.size()
            loss_er = torch.mean(torch.abs(cam1[:, 1:, :, :] - cam2[:, 1:, :, :]))

            cam1[:, 0, :, :] = 1 - torch.max(cam1[:, 1:, :, :], dim=1)[0]
            cam2[:, 0, :, :] = 1 - torch.max(cam2[:, 1:, :, :], dim=1)[0]

            tensor_ecr1 = torch.abs(max_onehot(cam2.detach()) - cam_rv1)  # *eq_mask
            tensor_ecr2 = torch.abs(max_onehot(cam1.detach()) - cam_rv2)  # *eq_mask
            loss_ecr1 = torch.mean(torch.topk(tensor_ecr1.view(ns, -1), k=(int)(21 * hs * ws * 0.2), dim=-1)[0])
            loss_ecr2 = torch.mean(torch.topk(tensor_ecr2.view(ns, -1), k=(int)(21 * hs * ws * 0.2), dim=-1)[0])
            loss_ecr = loss_ecr1 + loss_ecr2

            loss_cls = (loss_cls1 + loss_cls2) / 2 + (loss_rvmin1 + loss_rvmin2) / 2
            
            ##
            t1 = time()
            time_loss_cls_er_ecr = t2-t1
            ##
            ################################################################################
            ###################### Local Contrastive Learning ##############################
            ################################################################################
            loss_local_cross_nce1 = 0
            loss_local_cross_nce2 = 0
            loss_local_intra_nce = 0

            for i in range(f1_all_res.shape[0]):
                single_loss_local_cross_nce1, single_loss_local_cross_nce2, single_loss_local_intra_nce = loss_local(f1_all_res[i].unsqueeze(0), f2_all_res[i].unsqueeze(0))
                loss_local_cross_nce1 = loss_local_cross_nce1 + single_loss_local_cross_nce1
                loss_local_cross_nce2 = loss_local_cross_nce2 + single_loss_local_cross_nce2
                loss_local_intra_nce = loss_local_intra_nce + single_loss_local_intra_nce
            
            loss_local_cross_nce1 = loss_local_cross_nce1 / args.batch_size
            loss_local_cross_nce2 = loss_local_cross_nce2 / args.batch_size
            loss_local_intra_nce = loss_local_intra_nce / args.batch_size

            # loss_local_cross_nce1, loss_local_cross_nce2, loss_local_intra_nce = loss_local(f1_all_res, f2_all_res)
            ##
            t2 = time()
            time_loss_local = t1 -t2
            ##
            """""""""
                loss_local consist of 
                    cluster_prototype_generation + association
                    build_cross_pair
                    loss_cross
                    build_intra_pair
                    loss_intra
            """""""""

            ################################################################################
            ###################### Contrastive Learning ####################################
            ################################################################################
            
            ## class prototype generation
            f_proj1 = F.interpolate(f_proj1, size=(128 // 8, 128 // 8), mode='bilinear', align_corners=True)
            ## f_proj2 = F.interpolate(f_proj2, size=(128 // 8, 128 // 8), mode='bilinear', align_corners=True) --> 원래 사이즈가 16,16이므로 굳이 줄여줄 핋요 없음
            cam_rv1_down = F.interpolate(cam_rv1_down, size=(128 // 8, 128 // 8), mode='bilinear', align_corners=True)
            cam_rv2_down = cam_rv2_down

            prototypes1, pseudo_label1 = cls_proto_gen(f_proj1, cam_rv1_down, label, args.bg_threshold)
            prototypes2, pseudo_label2 = cls_proto_gen(f_proj1, cam_rv1_down, label, args.bg_threshold)

            ##
            t1 = time()
            time_class_proto_gen = t2-t1
            ##

            # 1. cross-view contrastive learning

            # for source
            n_f, c_f, h_f, w_f = f_proj1.shape
            f_proj1 = f_proj1.permute(0, 2, 3, 1).reshape(n_f * h_f * w_f, c_f)
            f_proj1 = F.normalize(f_proj1, dim=-1)
            pseudo_label1 = pseudo_label1.reshape(-1)
            positives1 = prototypes2[pseudo_label1]
            negatives1 = prototypes2

            # for target
            # n_f, c_f, h_f, w_f = f_proj2.shape ## f_proj1.shape == f_proj2.shape
            f_proj2 = f_proj2.permute(0, 2, 3, 1).reshape(n_f * h_f * w_f, c_f)
            f_proj2 = F.normalize(f_proj2, dim=-1)
            pseudo_label2 = pseudo_label2.reshape(-1)
            positives2 = prototypes1[pseudo_label2]
            negatives2 = prototypes1
            
            loss_cross_nce1, loss_cross_nce2 = loss_cross(f_proj1, f_proj2, positives1, negatives1, positives2, negatives2)

            ##
            t2 = time()
            time_loss_cross = t1 -t2
            ##

            # 2. intra-view contrastive learning

            # for source
            positives_intra1 = prototypes1[pseudo_label1]
            negatives_intra1 = prototypes1
            
            # for target
            positives_intra2 = prototypes2[pseudo_label2]
            negatives_intra2 = prototypes2

            num_cls = 21
            loss_intra_nce1 = loss_intra(f_proj1, positives_intra1, negatives_intra1, pseudo_label1, num_cls, (n_f, c_f, h_f, w_f))
            loss_intra_nce2 = loss_intra(f_proj2, positives_intra2, negatives_intra2, pseudo_label2, num_cls, (n_f, c_f, h_f, w_f))


            loss_intra_nce = 0.1 * (loss_intra_nce1 + loss_intra_nce2) / 2
            ##
            t1 = time()
            time_loss_intra = t2-t1
            ##
            # 3. total nce loss
            loss_nce = loss_cross_nce1 + loss_cross_nce2 + loss_intra_nce

            beta = 1
            loss_local_cross_nce1 *= beta
            loss_local_cross_nce2 *= beta
            loss_local_intra_nce *= beta
            loss_local_nce = loss_local_cross_nce1 + loss_local_cross_nce2 + loss_local_intra_nce
            
            # 4. total loss
            loss = loss_cls + loss_er + loss_ecr + loss_nce + loss_local_nce
            ##
            t2 = time()
            time_loss_total_sum = t1 -t2
            ##
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ##
            t1 = time()
            time_backward = t2-t1
            ##
            avg_meter.add({'loss': loss.item(),
                           'loss_cls': loss_cls.item(),
                           'loss_er': loss_er.item(),
                           'loss_ecr': loss_ecr.item(),
                           'loss_nce': loss_nce.item(),
                           'loss_intra_nce': loss_intra_nce.item(),
                           'loss_cross_nce': loss_cross_nce1.item(),
                           'loss_cross_nce2': loss_cross_nce2.item(),
                           'loss_local_nce': loss_local_nce.item(),
                           'loss_local_intra_nce':loss_local_intra_nce.item(),
                           'loss_local_cross_nce': loss_local_cross_nce1.item(),
                           'loss_local_cross_nce2': loss_local_cross_nce2.item()})

            if (optimizer.global_step - 1) % 5 == 0:
            # if (optimizer.global_step - 1) % 50 == 0:
                timer.update_progress(optimizer.global_step / max_step)

                print('Iter:%5d/%5d | ' % (optimizer.global_step - 1, max_step),
                      'loss: %.4f| loss_cls: %.4f| loss_er: %.4f| loss_ecr: %.4f| '
                      'loss_nce: %.4f| loss_intra_nce: %.4f| loss_cross_nce: %.4f| loss_cross_nce2: %.4f| loss_local_nce: %.4f| loss_local_intra_nce: %.4f| loss_local_cross_nce: %.4f| loss_local_cross_nce2: %.4f'
                      % avg_meter.get('loss', 'loss_cls', 'loss_er', 'loss_ecr', 'loss_nce', 'loss_intra_nce',
                                      'loss_cross_nce', 'loss_cross_nce2', 'loss_local_nce', 'loss_local_intra_nce','loss_local_cross_nce','loss_local_cross_nce2'),
                      'imps:%.1f | ' % ((iter + 1) * args.batch_size / timer.get_stage_elapsed()),
                      'Fin:%s | ' % (timer.str_est_finish()),
                      'lr: %.4f' % (optimizer.param_groups[0]['lr']), flush=True)

                avg_meter.pop()

                loss_dict = {'loss': loss.item(),
                             'loss_cls': loss_cls.item(),
                             'loss_er': loss_er.item(),
                             'loss_ecr': loss_ecr.item(),
                             'loss_nce': loss_nce.item(),
                             'loss_intra_nce': loss_intra_nce.item(),
                             'loss_cross_nce': loss_cross_nce1.item(),
                             'loss_cross_nce2': loss_cross_nce2.item(),
                             'loss_local_nce': loss_local_nce.item(),
                             'loss_local_intra_nce':loss_local_intra_nce.item(),
                             'loss_local_cross_nce': loss_local_cross_nce1.item(),
                             'loss_local_cross_nce2': loss_local_cross_nce2.item()}
                
                ## wandb
                # wandb.log({'loss': loss.item(),
                #            'loss_cls': loss_cls.item(),
                #            'loss_er': loss_er.item(),
                #            'loss_ecr': loss_ecr.item(),
                #            'loss_nce': loss_nce.item(),
                #            'loss_intra_nce': loss_intra_nce.item(),
                #            'loss_cross_nce': loss_cross_nce1.item(),
                #            'loss_cross_nce2': loss_cross_nce2.item(),
                #            'loss_local_nce': loss_local_nce.item(),
                #            'loss_local_intra_nce':loss_local_intra_nce.item(),
                #            'loss_local_cross_nce': loss_local_cross_nce1.item(),
                #            'loss_local_cross_nce2': loss_local_cross_nce2.item()})
                ##

                itr = optimizer.global_step - 1
                tblogger.add_scalars('loss', loss_dict, itr)
                tblogger.add_scalar('lr', optimizer.param_groups[0]['lr'], itr)
            ##
            t2 = time()
            time_logging = t1 -t2
            ##
            
            # wandb.log({'time_pre_process': time_pre_process,
            #            'time_forward1': time_forward1,
            #            'time_post_process1': time_post_process1,
            #            'time_forward2': time_forward2,
            #            'time_post_process2': time_post_process2,
            #            'time_loss_cls_er_ecr': time_loss_cls_er_ecr,
            #            'time_loss_local': time_loss_local,
            #            'time_class_proto_gen': time_class_proto_gen,
            #            'time_loss_cross': time_loss_cross,
            #            'time_loss_intra': time_loss_intra,
            #            'time_loss_total_sum': time_loss_total_sum,
            #            'time_backward': time_backward,
            #            'time_logging': time_logging})
        else:
            print('')
            timer.reset_stage()
            torch.save(model.module.state_dict(), "./pth/"+args.session_name + "_ep_" + str(ep) + '.pth')
    print(args.session_name)

    torch.save(model.module.state_dict(), "./pth/"+args.session_name + '.pth')
