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

#import torch.nn.functional as nn


from detectron2.config import configurable
from detectron2.layers import Conv2d, ConvTranspose2d, cat, interpolate
from detectron2.structures import Instances, heatmaps_to_keypoints
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry

_TOTAL_SKIPPED = 0


print('********************USING INTEGRAL INNOVATE SCRIPT *****************')

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

    #James softmax for per e.g ankle heatmap
    #tempheat = torch.reshape(heatmap, (heatmap.shape[0],heatmap.shape[1], -1))
    #print(tempheat.shape)
    #tempheat = torch.reshape(tempheat, (heatmap.shape[0]*heatmap.shape[1], -1))
    #print(tempheat.shape)
    #h_norm = nn.functional.softmax(tempheat.float(),1)

    #reshape back
    #h_norm = torch.reshape(h_norm, (heatmap.shape[0],heatmap.shape[1], -1))
    #h_norm = torch.reshape(h_norm, (heatmap.shape[0], heatmap.shape[1], heatmap.shape[2],heatmap.shape[3]))
    


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


    # start_x = 0.0
    # start_y = 8.637619

    # scale_x = 1 / (1236.8367 - 0.0)#bottom part of min-max normalization with division
    # scale_y = 1 / (619.60706 - 8.637619 )

    # scale_inv_x = (1236.8367 - 0.0) #bottom part of min-max normalization without division yet
    # scale_inv_y = (619.60706 - 8.637619 )

    # # all locations p in the domain, 
    # x_list = torch.linspace(0,1, h).cuda()
    # y_list = torch.linspace(0,1, w).cuda()

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
    i_ = i_ * scale_inv_x.reshape(-1,1) + start_x.reshape(-1,1)
    j_ = j_ * scale_inv_y.reshape(-1,1) + start_y.reshape(-1,1)

    #i_ = i_ * scale_inv_x + start_x
    #j_ = j_ * scale_inv_y + start_y
    #print('i_ (after) as global coordinates', i_[0])

    #
    #print('scale x, scale_inv_x : ', scale_x, scale_inv_x)

    #Modified arrangement
    pose  = torch.stack((i_,j_),dim=2) #[[i,i,i,,], #(N,K, 2)
                                       #[j,j,j,,,]]

    print('checking, is I and J well placed as x,y?', pose[0][0:2])
    print('min and max of I ', torch.min(i_), torch.max(i_))
    print('min and max of J', torch.min(j_), torch.max(j_))
  
    # partx = pose[:,:,0:1]
    # partx = partx * scale_inv_x + start_x

    # party = pose[:,:,1:2]
    # party = party * scale_inv_y + start_y

    # pose = torch.stack((partx,  party), dim = -2)
    # pose = pose.squeeze(-1) #(N,K,2)

    #return relative global coordinates
    #print('pose relative global coordinates', pose[0][0])
    return ({'probabilitymap': h_norm, 'pose_2d': pose}) #(N,K, 2)

def effective_2d_3d(pose2D_normalized):
	pred_pose3d = model2(pose2D_normalized.float())

	return pred_pose3d


def keypoint_rcnn_loss(pred_keypoint_logits, instances, normalizer, linermodel):
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
        #print('other fields:', instances_per_image.get_fields())
        #print('can we get image dim programmatically? :', instances_per_image.ke
        #############################################
	#pose3d_pts = instances_per_image.gt_pose3d.cuda()
        #pose3d_pts = pose3d_pts.reshape(pose3d_pts.shape[0],6,3)
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
	#p3d.append(pose3d_pts)

    if len(heatmaps):
        keypoint_targets = cat(heatmaps, dim=0) #single vector (GT heatmaps)
        valid = cat(valid, dim=0).to(dtype=torch.uint8) #single vector
        valid = torch.nonzero(valid).squeeze(1)

    try:
    	kps = torch.cat(kps)
    except:
    	print('kps', kps)
    	kps = torch.cat(kps)
    # torch.mean (in binary_cross_entropy_with_logits) doesn't
    # accept empty tensors, so handle it separately
    if len(heatmaps) == 0 or valid.numel() == 0:
        global _TOTAL_SKIPPED
        _TOTAL_SKIPPED += 1
        storage = get_event_storage()
        storage.put_scalar("kpts_num_skipped_batches", _TOTAL_SKIPPED, smoothing_hint=False)
        return pred_keypoint_logits.sum() * 0

    

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
    print('confirm shape after integral ', pred_integral['pose_2d'].shape)
    pred_integral_v1 = pred_integral['pose_2d'].view(N * K, -1)[valid]

    ############################################################

    pred_integral_v2 = pred_integral['pose_2d'].reshape(N, -1)
    print('input to linear pred_integral', pred_integral_v2.shape)

    ##Dont exclude any kps for 2nd model
    ##The 1st model should be invariant to bad keypoints, such that it predicts for missing kps
  #   pred_3d = linermodel(pred_integral_v2)
  #   print('output shape from linear pred_integral', pred_3d.shape)
  #   print('what pred pose3d looks like', pred_3d[0])
  #   print('what GT pose3d looks like', pose3d_gt[0])
  #   pose3d_gt = pose3d_pts.reshape(pose3d_pts.shape[0],-1)
    

  #   print('Is pose3d_gt (N,18)?', pose3d_gt.shape) #N,18

  #   #Normalize 3d GT by mean-std
  #   mean_3d, std_3d = (torch.Tensor([ 333.5211,  218.9238,  364.4432,  186.7393,  392.3470,  166.7051,
  #        -945.6299, -946.6586, -871.1463, -868.2529, -959.4473, -961.2425,
  #        1055.2781, 1052.3322,  673.6290,  670.1853,  292.7418,  296.1209]).cuda(),
 	# torch.Tensor([ 12.9435,  12.9282,  13.6145,  21.5151,  17.2765,  32.8399, 143.9522,
  #        143.2831, 192.1633, 199.3143, 165.5452, 174.0693, 182.4063, 182.2134,
  #        161.9945, 159.9102, 146.8468, 146.0199]).cuda())

  #   pose3d_gt = (pose3d_gt - mean_3d)/std_3d
  #   print('normalized 3d pose GT sample: ', pose3d_gt[0])

  #   #remove invalids
  #   pred_3d = pred_3d.view(-1, 3)[valid]
  #   pose3d_gt = pose3d_gt.view(-1, 3)[valid]
  #   print('new shapes: pred_3d, pose3d_gt', pred_3d.shape, pose3d_gt.shape)


    ##############################################################



    #pose3d_loss = torch.nn.functional.mse_loss(pred_3d, pose3d_gt)
    #print('pose3d_LOSS: ', pose3d_loss)

    #2D loss
    
    #s1, s2 = kps.shape[0], kps.shape[1] #shape
    #exclude invlaid
    #kps = kps.view(s1*s2, -1)[valid]

    #print('pred_integral, kps', pred_integral[:5], kps[:5])
    #pose2d_loss = torch.nn.functional.mse_loss(pred_integral, kps)
    #pose2d_loss = torch.sqrt(torch.mean((pred_integral - kps)**2))
    

    # #3D loss
    # p3d = torch.cat(p3d)
    # m1, m2 = p3d.shape[0], p3d.shape[1] #shape 
    # #exclude invlaid
    # kps = kps.view(m1*m2, -1)[valid]
    # kps = kps.view(N,K,-1)
    # #mean-std normalization for 3d targets
    # mean_3d = np.array([[  389.9240,   253.0210,   409.7404],
    #     [  232.3254,   427.8259,   225.9603],
    #     [-1170.1398, -1179.4377, -1179.5839],
    #     [-1199.9395, -1231.7988, -1242.5614],
    #     [ 1241.7291,  1239.4333,   841.8918],
    #     [  830.0914,   419.7809,   422.6696]])

    # std_3d = np.array([[22.3731, 10.8602, 14.1813],
    #     [14.6576, 26.9243, 22.4345],
    #     [40.7331, 37.2196, 38.8047],
    #     [38.7710, 40.0700, 38.5498],
    #     [27.9540, 25.8141, 19.8423],
    #     [13.2910,  2.9173,  3.5576]])

    # targets = (kps - mean_3d)/std_3d

    # #flattened vector for mse loss
    # targets = kps.view(N,-1)

    # #simple normalization 
    # #Min-max Normalization for 2D output using Full Image
    # pred_integral = pred_integral.view(N,K,-1)
    # xmax, xmin, ymax, ymin = 1236.8367, 0.0, 619.60706, 8.637619

    # partx = pred_integral[:,:,0:1]
    # partx = (partx - xmin)/(xmax - xmin)

    # party = kps[:,:,1:2]
    # party = (party - ymin)/(ymax - ymin)

    # pred_integral = torch.stack((partx,  party), dim = -2)
    # pred_integral = pred_integral.squeeze(-1) #(N,K,2)
    # #flattened vector
    # pose2D_normalized = (pred_integral.view(N, -1))*2-1 # bring it to -1...1
    # pred_pose3d = effective_2d_3d(pose2D_normalized)

    # print('3d pred integral output: ', pred_pose3d.shape, pred_pose3d[0[0]])
    # #pred_integral = pred_integral['pose_3d'].view(N * K, -1)[valid]

    # criterion = nn.MSELoss(size_average=True).cuda()
    # # calculate loss
    # #optimizer.zero_grad()
    # pose3d_loss = criterion(pred_pose3d, targets)
    # print('raw loss', pose3d_loss)
    # #losses.update(loss.item(), inputs.size(0))
    #loss.backward()
    #if max_norm:
    #nn.utils.clip_grad_norm(model2.parameters(), max_norm=1)
    #optimizer2.step()

    
    
    #print('pred_integral removed shape', pred_integral.shape)
    #kps = torch.cat(kps)

    #normalize kps
    
    # kp_mean = torch.Tensor([[942.8855, 326.6883],
    #     [941.4666, 405.1611],
    #     [740.3054, 304.9617],
    #     [737.7035, 421.5804],
    #     [530.7987, 290.6349],
    #     [534.2322, 425.0898]]).cuda()

    # kp_std = torch.Tensor([[ 94.6912,  31.1105],
    #     [ 96.2150,  31.2903],
    #     [ 89.2333,  28.6181],
    #     [ 89.7864,  32.5412],
    #     [109.8567,  45.1855],
    #     [ 92.0391,  33.6960]]).cuda()

    #All data mean-std normalization
    #kps = (kps - kp_mean)/kp_std

    # With Batch mean and std, you wont have fixed values to de-normarlize

    

    #Min-max Normalization using Full Image
    # xmax, xmin, ymax, ymin = 1236.8367, 0.0, 619.60706, 8.637619

    # partx = kps[:,:,0:1]
    # partx = (partx - xmin)/(xmax - xmin)

    # party = kps[:,:,1:2]
    # party = (party - ymin)/(ymax - ymin)

    # kps = torch.stack((partx,  party), dim = -2)
    # #print('1st kps', kps.shape)
    # kps = kps.squeeze(-1)


    #print('raw kps shape', kps.shape)
    #keypoint_loss = torch.nn.functional.mse_loss(pred_integral, keypoint_targets[valid])
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
    #print()
    #print('raw loss', pose2d_loss)

    #################################################
    #comb_loss = pose2d_loss *0.5 + pose3d_loss *0.5
    #print('comb_loss: ', comb_loss)

    # # keypoint_loss = F.cross_entropy(
    # #     pred_keypoint_logits[valid], keypoint_targets[valid], reduction="sum"
    # # )m

    # If a normalizer isn't specified, normalize by the number of visible keypoints in the minibatch
    if normalizer is None:
        normalizer = valid.numel()
    pose2d_loss /= normalizer

    
    print('normalized loss: ', pose2d_loss, 'normalizer amount: ', normalizer)
    print()
    return pose2d_loss


def keypoint_rcnn_inference(pred_keypoint_logits, pred_instances):
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

    out = integral_2d_innovate(pred_keypoint_logits)
    heatmap_norm = out['probabilitymap']
    print('heatmap_norm shape', heatmap_norm.shape)
    print('hip heatmap_norm', heatmap_norm[0][0][0])
    print('heatmap prob sum to 1: ', torch.sum(heatmap_norm[0][0]))
    #scores for the ankle etc
    scores = torch.max(torch.max(heatmap_norm, dim = -1)[0], dim = -1)[0]
    print('scores: ', scores)
    #max_ = torch.max(torch.max(heatmap, dim=-1)[0], dim=-1, keepdim=True)[0].unsqueeze(-1) #soving the numerical problem
    #unstack
    i_, j_  = torch.unbind(out['pose_2d'], dim=2)

    #de-normalize
    xmax, xmin, ymax, ymin = 1236.8367, 0.0, 619.60706, 8.637619
    i_ = (i_ * (xmax - xmin)) + xmin
    j_ = (j_ * (ymax - ymin)) + ymin

    #instance, K, 3) 3-> (x, y, score)
    keypoint_results = torch.stack((i_,j_, scores),dim=2)
    print('pred keypoint_results before split', keypoint_results.shape)
    num_instances_per_image = [len(i) for i in pred_instances]
    keypoint_results = keypoint_results[:, :, [0, 1, 3]].split(num_instances_per_image, dim=0)
    #print('pred keypoint_results after split', keypoint_results.shape)
    print('sample pred keypoint_results after split', keypoint_results[0][0])
    print('pred_instances', len(pred_instances))

    for keypoint_results_per_image, instances_per_image in zip(keypoint_results, pred_instances):
        # keypoint_results_per_image is (num instances)x(num keypoints)x(x, y, score)
        
        print('keypoint_results_per_image', keypoint_results_per_image.shape)
        #print('type:', instances_per_image.pred_keypoints.shape)
        instances_per_image.pred_keypoints = keypoint_results_per_image #.unsqueeze(0)
        
    #instances_per_image.pred_keypoints = keypoint_results

########################################################## MY CODE

def weight_init(m):
    if isinstance(m, nn.Linear):
        pass
        #nn.init.constant(m.bias, 0)
        #nn.init.kaiming_normal(m.weight)


class Linear(nn.Module):
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
                 num_stage=2,
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
        self.linermodel = LinearModel()
        #self.linermodel.apply(weight_init)

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
                "loss_keypoint": keypoint_rcnn_loss(x, instances, normalizer=normalizer, linermodel=self.linermodel)
                * self.loss_weight
            } #self.model2, self.optimizer2
        else:
            keypoint_rcnn_inference(x, instances)
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





