#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import List
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from detectron2.modeling.roi_heads import custom_plotting 
import matplotlib.pyplot as plt

#import utils
#import plotting
#import triangulate 
import numpy as np

#from PIL import Image
import pandas as pd
#import torch
#import torchvision
#import torchvision.transforms as transforms
#import h5py
import os
from IPython import display
import cv2
#import torch.nn.functional as nn


from detectron2.config import configurable
from detectron2.layers import Conv2d, ConvTranspose2d, cat, interpolate
from detectron2.structures import Instances, heatmaps_to_keypoints
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry

_TOTAL_SKIPPED = 0
_TOTAL_SKIPPED_KPS = 0
_LOSSES_2D, _LOSSES_3D, _LOSSES_COMB = [], [], []
_PCK_SCORE = 0

print('********************USING END-END SCRIPT *****************')

__all__ = [
    "ROI_KEYPOINT_HEAD_REGISTRY",
    "build_keypoint_head",
    "BaseKeypointRCNNHead",
    "KRCNNConvDeconvUpsampleHead",
]


ROI_KEYPOINT_HEAD_REGISTRY = Registry("ROI_KEYPOINT_HEAD")
ROI_KEYPOINT_HEAD_REGISTRY.__doc__ = """
Registry for keypoint heads, which make keypoint predictions from per-region features.
The registered object will be called with `obj(cfg, input_shape)`.
"""

import torch.nn as nn


def build_keypoint_head(cfg, input_shape):
    """
    Build a keypoint head from `cfg.MODEL.ROI_KEYPOINT_HEAD.NAME`.
    """
    name = cfg.MODEL.ROI_KEYPOINT_HEAD.NAME
    return ROI_KEYPOINT_HEAD_REGISTRY.get(name)(cfg, input_shape)

def integral_2d_innovate(heatmap, rois):
    #print('2d Innovate being used')
    #heatmap i.e pred_keypoint_logits (Tensor): A tensor of shape (N, K, S, S) / (N, K, H, W) 
    h, w = heatmap.shape[2], heatmap.shape[3]
    #print('origin logits bf heatmap', heatmap.shape)

    #implementing softmax (this was for a batch)
    #softmax -max soln works even with neg values
    max_ = torch.max(torch.max(heatmap, dim=-1)[0], dim=-1, keepdim=True)[0].unsqueeze(-1) #soving the numerical problem
    print('max shape', max_.shape)
    heatmap = heatmap - max_

    exp_heatmap = torch.exp(heatmap)
    h_norm = exp_heatmap / torch.sum(exp_heatmap, dim = (-1,-2), keepdim = True)

    #Any NAN in hnorm
    test = h_norm.cpu().detach().clone()
    test = test.numpy()
    print('HNORM contains nan ?:', ['YES' if np.sum(np.isnan(test)) else 'NO'])

    #DISCRETE FORM of the Integral Equation
    # computing integral in relative global coordinates directly

    #print('rois in integral function ', rois)
    start_x = rois[:, 0]
    start_y = rois[:, 1]

    scale_x = 1 / (rois[:, 2] - rois[:, 0])#bottom part of min-max normalization with division
    scale_y = 1 / (rois[:, 3] - rois[:, 1])

    scale_inv_x = (rois[:, 2] - rois[:, 0]) #bottom part of min-max normalization without division yet
    scale_inv_y = (rois[:, 3] - rois[:, 1])

    #DISCRETE FORM of integral is not expensive 
    #Our choice (0->1) ROI coordinates 
    x_list = torch.linspace(0,1, w).cuda()
    y_list = torch.linspace(0,1, h).cuda()
    # 3D Heatmap z_list = torch.linspace(0,1,z).cuda()
    i,j = torch.meshgrid(x_list, y_list)

    #weighted by their probabilities.
    i_ = torch.sum(i*h_norm, dim=(-1,-2))
    j_ = torch.sum(j*h_norm, dim=(-1,-2))

    # transforming back to global relative coords
    #print('i_, scale_inv_x, start_x', i_.shape, scale_inv_x, start_x)
    print('i_ (before) as 0-1 coordinates', i_[0])
    i_g = i_ * scale_inv_x.reshape(-1,1) + start_x.reshape(-1,1)
    j_g = j_ * scale_inv_y.reshape(-1,1) + start_y.reshape(-1,1)

    #Modified arrangement
    pose_glob  = torch.stack((i_g,j_g),dim=2) #[[i,i,i,,], #(N,K, 2)
                                       #[j,j,j,,,]]

    pose_norm = torch.stack((i_,j_),dim=2) #[[i,i,i,,], #(N,K, 2)
                                       #[j,j,j,,,]]


    print('checking, is I and J well placed as x,y?', pose_glob[0][0:2])
    print('min and max of I ', torch.min(i_), torch.max(i_))
    print('min and max of J', torch.min(j_), torch.max(j_))

    #return relative global coordinates
    #print('pose relative global coordinates', pose[0][0])
    return ({'probabilitymap': h_norm, 'pose_2d global': pose_glob, 'pose_2d norm': pose_norm,}) #(N,K, 2)


def keypoint_rcnn_loss(pred_keypoint_logits, instances, normalizer, linearmodel):
    """
    Arguments:
        pred_keypoint_logits (Tensor): A tensor of shape (N, K, S, S) where N is the total number
            of instances in the batch, K is the number of keypoints, and S is the side length
            of the keypoint heatmap. The values are spatial logits.
            predicted keypoint heatmaps in `pred_keypoint_logits`

        instances (list[Instances]): A list of M Instances, where M is the batch size.
            These instances are predictions from the model
            that are in 1:1 correspondence with pred_keypoint_logits.
            Each Instances should contain a `gt_keypoints` field containing a `structures.Keypoint`
            instance.
        normalizer (float): Normalize the loss by this amount.
            If not specified, we normalize by the number of visible keypoints in the minibatch.
    Returns a scalar tensor containing the loss.
    """

    heatmaps = []
    valid = []
    kps = []
    p3d = []

    color_order_ego = [1, 3, 5, 7, 2, 4, 6,];

    bones_ego = [[0,1], [0,2],[2,4],[1,3], [3,5]]

    N, K, H, W = pred_keypoint_logits.shape
    keypoint_side_len = pred_keypoint_logits.shape[2]



    # flatten all GT bboxes from all images together (list[Boxes] -> Rx4 tensor)
    print('check for box rois: ', [b.proposal_boxes.tensor for b in instances])
    bboxes_flat = cat([b.proposal_boxes.tensor for b in instances], dim=0)
    rois = bboxes_flat.detach()

    #M = len(instances)
    #kps =  torch.zeros(M, )
    for instances_per_image in instances:
        if len(instances_per_image) == 0:
            continue
        keypoints = instances_per_image.gt_keypoints
        # if len(keypoints) ==0:
        #   print('EMPTY KEYPOINTS, WHY?') 
        #   continue


        #print('other fields:', instances_per_image.get_fields())
        #print('can we get image dim programmatically? :', instances_per_image.ke
        #############################################
        pose3d_pts = instances_per_image.gt_pose3d.cuda()
        pose3d_pts = pose3d_pts.reshape(pose3d_pts.shape[0],6,3)
        
        ############################################################
        #e.g (8,6,3)
        #print('Daniel test keypoints', keypoints.tensor.shape)
        #GT keypoints -> GT heatmaps  
        heatmaps_per_image, valid_per_image = keypoints.to_heatmap(
            instances_per_image.proposal_boxes.tensor, keypoint_side_len
        )

        #print('keypoint 2 GT heatmap => Indices of ROI, lets see hip heatmap', heatmaps_per_image.shape, heatmaps_per_image[0][0])
        #GT heatmaps -> to 1D vector
        heatmaps.append(heatmaps_per_image.view(-1)) #N*K
        valid.append(valid_per_image.view(-1)) #stretch to 1D vector
        #print('keypoints.tensor[:,:,0:2]', keypoints.tensor[:,:,0:2].shape)
        kps.append(keypoints.tensor[:,:,0:2]) #exclude visibility out
        ###################################
        p3d.append(pose3d_pts)

    if len(heatmaps):
        keypoint_targets = cat(heatmaps, dim=0) #single vector (GT heatmaps)
        valid = cat(valid, dim=0).to(dtype=torch.uint8) #single vector
        valid = torch.nonzero(valid).squeeze(1)


    # torch.mean (in binary_cross_entropy_with_logits) doesn't
    # accept empty tensors, so handle it separately
    if len(heatmaps) == 0 or valid.numel() == 0:
        global _TOTAL_SKIPPED
        _TOTAL_SKIPPED += 1
        storage = get_event_storage()
        storage.put_scalar("kpts_num_skipped_batches", _TOTAL_SKIPPED, smoothing_hint=False)
        return pred_keypoint_logits.sum() * 0

    
    kps = torch.cat(kps)
    p3d = torch.cat(p3d)
    print('GT pose2d shape', kps.shape)
    print('GT pose3d shape', p3d.shape)

    keep_kps = kps

    print('min and max of pred_keypoint_logits', torch.min(pred_keypoint_logits), torch.max(pred_keypoint_logits))
    # pred_keypoint_logits = pred_keypoint_logits.view(N * K, H * W)
    # pred_keypoint_logits_  = pred_keypoint_logits[valid].view(N,K, H,W)
    #pred_keypoint_logits = pred_keypoint_logits.view(N * K, H * W)

    #lets confirm equal total instances
    try:
        assert (kps.shape[0] == pred_keypoint_logits.shape[0])
    except:
        print('kps shape', kps.shape, 'pred_keypoint_logits shape', pred_keypoint_logits.shape)
        assert (kps.shape[0] == pred_keypoint_logits.shape[0])

    # if use_2d:
    #print('pred_keypoint_logits', pred_keypoint_logits[0][0:2])
    #print('using 2d innovate')
    #print('raw pred_keypoint_logits', pred_keypoint_logits.shape)
    pred_integral = integral_2d_innovate(pred_keypoint_logits, rois)
    print('confirm shape after integral ', pred_integral['pose_2d global'].shape)
    pred_integral_v1 = pred_integral['pose_2d global'].view(N * K, -1)[valid]


    s1, s2 = kps.shape[0], kps.shape[1] #shape
    print('kps shape before removing invalid', kps.shape)
    kps = kps.view(s1*s2, -1)[valid]
    print('kps removed invalid shape', kps.shape)


    print('example pred: ', pred_integral_v1[-3:])
    print('example kps: ', kps[-3:])
    print()
    #print('final kps shape',kps.shape, 'final pred shape', pred_integral.shape)
    print('min and max of pred_integral_v1', torch.min(pred_integral_v1), torch.max(pred_integral_v1))
    print('min and max of kps', torch.min(kps), torch.max(kps))

    pose2d_loss = torch.nn.functional.mse_loss(pred_integral_v1, kps, reduction = 'sum')
    print('original pose2d loss ', pose2d_loss)

    if normalizer is None:
        normalizer = valid.numel()
    pose2d_loss /= normalizer

    
    #
    my_normalizer= 720 + 1280 
    pose2d_loss /= my_normalizer


    ############################################################

    pred_integral_v2 = pred_integral['pose_2d global'].reshape(N, -1)
    print('input to linear pred_integral', pred_integral_v2.shape)

    ##Dont exclude any kps for 2nd model
    ##The 1st model should be invariant to bad keypoints, such that it predicts for missing kps
    

    #pred_3d = linearmodel(keep_kps.view(keep_kps.shape[0], -1)) #(1,18)
    pred_3d = linearmodel(pred_integral_v2) #(1,18)

    # try:
    #     pred_3d = linearmodel(pred_integral_v2)
    #     print('Another linear model worked..No error')
    # except:
    #     pass

    print('output shape from linear pred_integral', pred_3d.shape)
    print('what pred pose3d looks like', pred_3d[0])
    pose3d_gt = p3d.reshape(p3d.shape[0],-1) #N, 18

    #pose3d_gt = pose3d_gt[0].unsqueeze(0) #(1,18) pick only 1 since they are duplicates
    print('what GT pose3d looks like', pose3d_gt[0])
    
    

    

    #Normalize 3d GT by mean-std
  #   mean_3d, std_3d = (torch.Tensor([ 333.5211,  218.9238,  364.4432,  186.7393,  392.3470,  166.7051,
  #        -945.6299, -946.6586, -871.1463, -868.2529, -959.4473, -961.2425,
  #        1055.2781, 1052.3322,  673.6290,  670.1853,  292.7418,  296.1209]).cuda(),
    # torch.Tensor([ 12.9435,  12.9282,  13.6145,  21.5151,  17.2765,  32.8399, 143.9522,
  #        143.2831, 192.1633, 199.3143, 165.5452, 174.0693, 182.4063, 182.2134,
  #        161.9945, 159.9102, 146.8468, 146.0199]).cuda())


    #Normalize relative to the hip
    pose3d_gt = pose3d_gt.view(pose3d_gt.shape[0], 6,3) #N,6,3


    midhip = (pose3d_gt[:,0] + pose3d_gt[:,1])/2

    print('pose3d_gt shape, midhip shape', pose3d_gt.shape, midhip.unsqueeze(1).shape)
    pose3d_gt = pose3d_gt - midhip.unsqueeze(1)
    pose3d_gt = pose3d_gt.view(pose3d_gt.shape[0], -1)

    pose3d_gt_raw = pose3d_gt


    print('Is pose3d_gt (1,18)?', pose3d_gt.shape) #1,18

    #Normalize 3d GT by mean-std relative to the hip (Project 2)
    # mean_3d, std_3d = (torch.Tensor([   90.4226,   -99.0404,   113.7033,   -90.4226,    99.0404,  -113.7033,
    #     -1257.6155, -1297.9100, -1227.4360, -1220.5818, -1329.1154, -1301.5215,
    #       797.3640,   756.3050,   403.3004,   410.9879,   -14.6912,    16.2920]).cuda(),
    # torch.Tensor([ 15.5230,  19.4742,  25.6194,  15.5230,  19.4742,  25.6194, 183.8460,
    #     172.6190, 212.3050, 218.0117, 192.0247, 208.0867, 178.1015, 186.4496,
    #     160.7282, 160.8192, 163.5823, 152.6740]).cuda())

    #Normalize 3d GT by mean-std relative to the hip (Project 3)
    mean_3d, std_3d = (torch.Tensor([   90.4712,  -100.0514,   113.9451,   -90.4712,   100.0514,  -113.9451,
        -1272.1144, -1312.7323, -1242.2559, -1237.8091, -1341.8601, -1318.6337,
          801.4175,   760.1226,   408.5650,   416.2698,    -7.5783,    23.3722]).cuda(),
    torch.Tensor([ 15.1813,  18.4730,  25.2961,  15.1813,  18.4730,  25.2961, 182.1320,
        169.5995, 213.0917, 218.8727, 193.2803, 208.0172, 176.0335, 183.7734,
        158.1118, 158.2999, 160.9153, 149.7474]).cuda())

    pose3d_gt = (pose3d_gt - mean_3d)/std_3d
    print('normalized 3d pose GT sample: ', pose3d_gt[0])

    pred_3d_star = pred_3d.view(-1, 3)
    pose3d_gt_star = pose3d_gt.view(-1, 3)
        #print('invalid removed, new shapes: pred_3d, pose3d_gt',type(pred_3d), type(pose3d_gt),pred_3d.shape, pose3d_gt.shape)
    
        #wxclude nans from GT pose3d]
    all_nan = torch.isnan(pose3d_gt_star)

    #consider all valid
    pose3d_loss = torch.nn.functional.mse_loss(pred_3d_star[~all_nan], pose3d_gt_star[~all_nan])
    # try:
    #   print('pose3d_LOSS: ', pose3d_loss)
    # except:
    #   print('pose3d_loss', torch.nn.functional.mse_loss(pred_3d, pose3d_gt))

    ##############################################################

    #comb_loss = pose2d_loss*1.0 + pose3d_loss*0.30 # score 0.6983
    #comb_loss = pose2d_loss*0.8 + pose3d_loss*0.30 # score 0.5844
    #comb_loss = pose2d_loss*0.70 + pose3d_loss*0.70
    #comb_loss = pose2d_loss*1.0 + pose3d_loss*3.0

    #pose2d_loss = 0
    comb_loss = pose2d_loss*0.70 + pose3d_loss*0.30 
    #comb_loss = pose3d_loss

    

    global _LOSSES_2D, _LOSSES_3D, _LOSSES_COMB
    _LOSSES_2D.append(pose2d_loss)
    _LOSSES_3D.append(pose3d_loss)
    _LOSSES_COMB.append(comb_loss)

    storage = get_event_storage()  
    print('storage', storage)
    #storage.p     
    #storage.put_scalar("comb_loss", _LOSSES)

    print('normalized loss: ', pose2d_loss, 'normalizer amount: ', normalizer)
    print('pose3d_LOSS: ', pose3d_loss)
    print('combined_loss: ', comb_loss)
    
    print()
    print()
    print()
    print()
    print()

    # # plot progress
    #only display if pose 3d GT has no nans 
    if np.sum(np.isnan(pose3d_gt.detach().cpu().numpy())) == 0 :
    #if 0:
        # clear figures for a new update
        fig=plt.figure(figsize=(20, 5), dpi= 80, facecolor='w', edgecolor='k')
        axes=fig.subplots(1,5)

        axs=[]
        f = plt.figure(figsize=(10,10))
        axs.append(f.add_subplot(2,3,1, projection='3d'))
        axs.append(f.add_subplot(2,3,2, projection='3d'))
        axs.append(f.add_subplot(2,3,3, projection='3d'))
        #axs.append(f.add_subplot(2,3,4, projection='3d'))
        #axs.append(f.add_subplot(2,3,5, projection='3d'))
        #axs.append(f.add_subplot(2,3,6, projection='3d'))


        #un-normalize for display 3D
        #pose3d_gt = (pose3d_gt * std_3d) + mean_3d
        pred_3d = (pred_3d * std_3d) + mean_3d
        pred_3d = pred_3d.view(pred_3d.shape[0], 6,3)

        #custom_plotting.plot_2Dpose(axs[0], pose3d_gt[0].detach().cpu().T,  bones=bones_ego, color_order=color_order_ego,flip_yz=False)
        #custom_plotting.plot_2Dpose(axs[0], pose3d_gt[0].detach().cpu().T,  bones=bones_ego, color_order=color_order_ego,flip_yz=False)

        img = '/content/drive/My Drive/black_img2.png'
        img = cv2.imread(img)
        img = cv2.resize(img, (720, 1280))
        img = np.array(img)
        keep_kps = keep_kps.view(keep_kps.shape[0], 6,2)

        
        custom_plotting.plot_3Dpose(axs[0], pose3d_gt_raw[0].detach().cpu(),  bones=bones_ego, color_order=color_order_ego,flip_yz=False)
        #custom_plotting.plot_3Dpose(axs[1], pose3d_gt_raw[1].detach().cpu(),  bones=bones_ego, color_order=color_order_ego,flip_yz=False)
        #custom_plotting.plot_3Dpose(axs[2], pose3d_gt_raw[2].detach().cpu(),  bones=bones_ego, color_order=color_order_ego,flip_yz=False)
        custom_plotting.plot_3Dpose(axs[1], pred_3d[0].detach().cpu(),  bones=bones_ego, color_order=color_order_ego,flip_yz=False)
        custom_plotting.plot_3Dpose(axs[2], pred_3d[1].detach().cpu(),  bones=bones_ego, color_order=color_order_ego,flip_yz=False)
        #custom_plotting.plot_3Dpose(axs[5], pred_3d[2].detach().cpu(),  bones=bones_ego, color_order=color_order_ego,flip_yz=False)


        custom_plotting.plotPoseOnImage(keep_kps[0].detach().cpu(), img, ax=axes[3])
        custom_plotting.plotPoseOnImage(pred_integral_v2.view(-1,6,2)[0].detach().cpu().float(), img, ax=axes[4])
        

        axes[0].plot(_LOSSES_2D)
        axes[0].set_yscale('log')
        # clear output window and diplay updated figure
        axes[1].plot(_LOSSES_3D)
        axes[1].set_yscale('log')

        axes[2].plot(_LOSSES_COMB)
        axes[2].set_yscale('log')

        display.clear_output(wait=True)
        #display.display(plt.gcf())
        plt.show()
        #plt.show()
        plt.close()
        #display.display()
        #print("Epoch {}, iteration {} of {} ({} %), loss={}".format(e, i, len(train_loader), 100*i//len(train_loader), losses[-1]))

    return comb_loss


def pck(target, pred, treshold=100):
    '''
    Percentage of Correct Keypoint for 3D pose Evaluation where PCKh @ 0.1m (10cm)

    Arguments:
    target: A tensor of shape (1, 18) : normalized values relative to hip
    pred: A tensor of shape (1, 18) : normalized values relative to hippck
    threshold87y 

    Returns:
        pck_score: A scalar value btw 0 and 1
    '''
    diff = torch.abs(target - pred)
    count = torch.sum(diff < treshold, dtype=torch.float)
    pck_score = count/ (target.shape[0]*target.shape[1])
    return pck_score



def keypoint_rcnn_inference(pred_keypoint_logits, pred_instances, linearmodel):
    """
    Post process each predicted keypoint heatmap in `pred_keypoint_logits` into (x, y, score)
        and add it to the `pred_instances` as a `pred_keypoints` field.
    Args:
        pred_keypoint_logits (Tensor): A tensor of shape (R, K, S, S) where R is the total number
           of instances in the batch, K is the number of keypoints, and S is the side length of
           the keypoint heatmap. The values are spatial logits.
        pred_instances (list[Instances]): A list of N Instances, where N is the number of images.
    Returns:
        None. Each element in pred_instances will contain an extra "pred_keypoints" field.
            The field is a tensor of shape (#instance, K, 3) where the last
            dimension corresponds to (x, y, score).
            The scores are larger than 0.
    """
    # flatten all bboxes from all images together (list[Boxes] -> Rx4 tensor)
    #bboxes_flat = cat([b.pred_boxes.tensor for b in pred_instances], dim=0)

    #keypoint_results = heatmaps_to_keypoints(pred_keypoint_logits.detach(), bboxes_flat.detach())
    #num_instances_per_image = [len(i) for i in pred_instances]
    #keypoint_results = keypoint_results[:, :, [0, 1, 3]].split(num_instances_per_image, dim=0)


    if pred_keypoint_logits.shape[0] == 0 :
        return None

    # flatten all GT bboxes from all images together (list[Boxes] -> Rx4 tensor)
    print('check for box rois inference: ', [b for i, b in enumerate(pred_instances) if i < 3])
    bboxes_flat = cat([b.pred_boxes.tensor for b in pred_instances], dim=0)
    pred_rois = bboxes_flat.detach()

    ###  2D   #####

    out = integral_2d_innovate(pred_keypoint_logits, pred_rois)
    heatmap_norm = out['probabilitymap']
    print('heatmap_norm shape', heatmap_norm.shape)
    print('hip heatmap_norm', heatmap_norm[0][0][0])
    print('heatmap prob sum to 1: ', torch.sum(heatmap_norm[0][0]))
    #scores for the ankle etc
    scores = torch.max(torch.max(heatmap_norm, dim = -1)[0], dim = -1)[0]
    #print('scores: ', scores)
    #max_ = torch.max(torch.max(heatmap, dim=-1)[0], dim=-1, keepdim=True)[0].unsqueeze(-1) #soving the numerical problem
    #print("type of out['pose_2d']", type(out['pose_2d']))
    #unstack

    i_, j_  = torch.unbind(out['pose_2d global'], dim=2)

    #instance, K, 3) 3-> (x, y, score)
    keypoint_results = torch.stack((i_,j_, scores),dim=2)
    keypoint_results_prev = heatmaps_to_keypoints(pred_keypoint_logits.detach(), bboxes_flat.detach())

    #print('pred keypoint_results before split', keypoint_results.shape)
    num_instances_per_image = [len(i) for i in pred_instances]
    # 0 for x, 1 for y, 3 for scores in heatmaps_to_keypoints function
    keypoint_results = keypoint_results[:, :, :].split(num_instances_per_image, dim=0)
    keypoint_results_prev = keypoint_results_prev[:, :, [0, 1, 3]].split(num_instances_per_image, dim=0)


    ###  3D   #####

    input2d = out['pose_2d global'].view(out['pose_2d global'].shape[0],-1)
    print('input 2d shape for testing', input2d.shape)

    # print('linearmodel.is_cuda ? ', next(linearmodel.parameters()).is_cuda)
    # print('input2d.is_cuda ? ', input2d.is_cuda)


    #print('type input2d', type(input2d))
    #print('input2d',input2d)
    print('min and max of input2d for testing', torch.min(input2d), torch.max(input2d))
    pred_3d = linearmodel(input2d)
    print('output 3d shape in testing', pred_3d.shape)
    print('min and max of out3d in testing', torch.min(pred_3d), torch.max(pred_3d))
    pred_3d  = pred_3d[:, :].split(num_instances_per_image, dim=0)


    cnt = 0
    for keypoint_results_per_image1,keypoint_results_per_image2, instances_per_image, pred_3d_results_per_image in zip(keypoint_results,keypoint_results_prev, pred_instances, pred_3d):
        # keypoint_results_per_image is (num instances)x(num keypoints)x(x, y, score)
        
        print('keypoint_results_per_image1', keypoint_results_per_image1.shape)
        print('keypoint_results_per_image2', keypoint_results_per_image2.shape)
        print('pred_3d_results_per_image', pred_3d_results_per_image.shape)

        print('scores from keypoint_results_per_image1: ', keypoint_results_per_image1[0, :, 2])
        print('scores from keypoint_results_per_image2: ', keypoint_results_per_image2[0, :, 2])

        keypoint_results_per_image1[:, :, 2] = keypoint_results_per_image2[:, :, 2] 

        instances_per_image.pred_keypoints = keypoint_results_per_image1#.unsqueeze(0)
        instances_per_image.pred_3d_pts = pred_3d_results_per_image #.unsqueeze(0)
        print('pred_3d_results_per_image sample', pred_3d_results_per_image[0])
        cnt+=1

    print('pred_instances length', cnt)


########################################################## MY CODE

def weight_init(m):
    if isinstance(m, nn.Linear):
        #pass
        #nn.init.constant(m.bias, 0)
        nn.init.kaiming_normal_(m.weight)
        #nn.init.uniform_(m.weight, 0, 1)


class Linear(nn.Module):
    ''' Represents the diagram in the paper for x1'''
    def __init__(self, linear_size, p_dropout=0.5):
        super(Linear, self).__init__()
        self.l_size = linear_size

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p_dropout)

        self.w1 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm1 = nn.BatchNorm1d(self.l_size)

        self.w2 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm2 = nn.BatchNorm1d(self.l_size)

    def forward(self, x):
        y = self.w1(x)
        y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)

        y = self.w2(y)
        y = self.batch_norm2(y)
        y = self.relu(y)
        y = self.dropout(y)

        out = x + y

        return out


class LinearModel(nn.Module):
    def __init__(self,
                 linear_size=1024,
                 num_stage=3,
                 p_dropout=0.5):
        super(LinearModel, self).__init__() 

        self.linear_size = linear_size
        self.p_dropout = p_dropout
        self.num_stage = num_stage

        # 2d joints
        self.input_size =  6 * 2
        # 3d joints
        self.output_size = 6 * 3

        # process input to linear size
        self.w1 = nn.Linear(self.input_size, self.linear_size)
        self.batch_norm1 = nn.BatchNorm1d(self.linear_size)

        self.linear_stages = []
        for l in range(num_stage):
            self.linear_stages.append(Linear(self.linear_size, self.p_dropout))
        self.linear_stages = nn.ModuleList(self.linear_stages)

        # post processing
        self.w2 = nn.Linear(self.linear_size, self.output_size)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(self.p_dropout)

    def forward(self, x):
        # pre-processing
        y = self.w1(x)
        y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)

        # linear layers
        for i in range(self.num_stage):
            y = self.linear_stages[i](y)

        y = self.w2(y)

        return y
##################################### END of MYCODE

class BaseKeypointRCNNHead(nn.Module):
    """
    Implement the basic Keypoint R-CNN losses and inference logic described in :paper:`Mask R-CNN`.
    """

    @configurable
    def __init__(self, *, num_keypoints, loss_weight=1.0, loss_normalizer=1.0):
        """
        NOTE: this interface is experimental.
        Args:
            num_keypoints (int): number of keypoints to predict
            loss_weight (float): weight to multiple on the keypoint loss
            loss_normalizer (float or str):
                If float, divide the loss by `loss_normalizer * #images`.
                If 'visible', the loss is normalized by the total number of
                visible keypoints across images.
        """
        super().__init__()
        self.num_keypoints = num_keypoints
        self.loss_weight = loss_weight
        assert loss_normalizer == "visible" or isinstance(loss_normalizer, float), loss_normalizer
        self.loss_normalizer = loss_normalizer
        self.linearmodel = LinearModel()
        try:
            print('using torch.device("cuda:0")')
            self.linearmodel = self.linearmodel.to(torch.device("cuda:0"))
        except:
            print('using cuda directly')
            self.linearmodel = self.linearmodel.cuda()
        
        self.linearmodel.apply(weight_init)
        print(">>> total params: {:.2f}M".format(sum(p.numel() for p in self.linearmodel.parameters()) / 1000000.0))
        
    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = {
            "loss_weight": cfg.MODEL.ROI_KEYPOINT_HEAD.LOSS_WEIGHT,
            "num_keypoints": cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS,
        }

        #2nd model
        #self.model2 = cfg.model2
        #self.optimizer2 = cfg.optimizer2

        normalize_by_visible = (
            cfg.MODEL.ROI_KEYPOINT_HEAD.NORMALIZE_LOSS_BY_VISIBLE_KEYPOINTS
        )  # noqa
        if not normalize_by_visible:
            batch_size_per_image = cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE
            positive_sample_fraction = cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION
            ret["loss_normalizer"] = (
                ret["num_keypoints"] * batch_size_per_image * positive_sample_fraction
            )
        else:
            ret["loss_normalizer"] = "visible"
        return ret

    def forward(self, x, instances: List[Instances]):
        """
        Args:
            x: input region feature(s) provided by :class:`ROIHeads`.
            instances (list[Instances]): contains the boxes & labels corresponding
                to the input features.
                Exact format is up to its caller to decide.
                Typically, this is the foreground instances in training, with
                "proposal_boxes" field and other gt annotations.
                In inference, it contains boxes that are already predicted.
        Returns:
            A dict of losses if in training. The predicted "instances" if in inference.
        """
        x = self.layers(x)
        if self.training:
            num_images = len(instances)
            normalizer = (
                None if self.loss_normalizer == "visible" else num_images * self.loss_normalizer
            )
            return {
                "loss_keypoint": keypoint_rcnn_loss(x, instances, normalizer=normalizer, linearmodel=self.linearmodel)
                * self.loss_weight
            } #self.model2, self.optimizer2
        else:
            keypoint_rcnn_inference(x, instances, linearmodel=self.linearmodel)
            return instances

    def layers(self, x):
        """
        Neural network layers that makes predictions from regional input features.
        """
        raise NotImplementedError


@ROI_KEYPOINT_HEAD_REGISTRY.register()
class KRCNNConvDeconvUpsampleHead(BaseKeypointRCNNHead):
    """
    A standard keypoint head containing a series of 3x3 convs, followed by
    a transpose convolution and bilinear interpolation for upsampling.
    """

    @configurable
    def __init__(self, input_shape, *, num_keypoints, conv_dims, **kwargs):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape (ShapeSpec): shape of the input feature
            conv_dims: an iterable of output channel counts for each conv in the head
                         e.g. (512, 512, 512) for three convs outputting 512 channels.
        """
        super().__init__(num_keypoints=num_keypoints, **kwargs)

        # default up_scale to 2 (this can be made an option)
        up_scale = 2
        in_channels = input_shape.channels

        self.blocks = []
        for idx, layer_channels in enumerate(conv_dims, 1):
            module = Conv2d(in_channels, layer_channels, 3, stride=1, padding=1)
            self.add_module("conv_fcn{}".format(idx), module)
            self.blocks.append(module)
            in_channels = layer_channels

        deconv_kernel = 4
        self.score_lowres = ConvTranspose2d(
            in_channels, num_keypoints, deconv_kernel, stride=2, padding=deconv_kernel // 2 - 1
        )
        self.up_scale = up_scale

        for name, param in self.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                pass
                # Caffe2 implementation uses MSRAFill, which in fact
                # corresponds to kaiming_normal_ in PyTorch
                #nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")
                #nn.init.uniform_(param, 0, 1)

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        ret["input_shape"] = input_shape
        ret["conv_dims"] = cfg.MODEL.ROI_KEYPOINT_HEAD.CONV_DIMS
        return ret

    def layers(self, x):
        for layer in self.blocks:
            x = F.relu(layer(x))
        x = self.score_lowres(x)
        x = interpolate(x, scale_factor=self.up_scale, mode="bilinear", align_corners=False)
        return x

