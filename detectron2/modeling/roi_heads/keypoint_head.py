# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import List
import torch
from torch import nn
from torch.nn import functional as F

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


def build_keypoint_head(cfg, input_shape):
    """
    Build a keypoint head from `cfg.MODEL.ROI_KEYPOINT_HEAD.NAME`.
    """
    name = cfg.MODEL.ROI_KEYPOINT_HEAD.NAME
    return ROI_KEYPOINT_HEAD_REGISTRY.get(name)(cfg, input_shape)

def integral_2d_innovate(heatmap):
    #print('2d Innovate being used')
    #heatmap i.e pred_keypoint_logits (Tensor): A tensor of shape (N, K, S, S) / (N, K, H, W) 
    h, w = heatmap.shape[2], heatmap.shape[3]

     #implementing softmax 
    heatmap = heatmap - torch.max(torch.max(heatmap, dim=-1)[0], dim=-1, keepdim=True)[0].unsqueeze(-1) #soving the numerical problem
    exp_heatmap = torch.exp(heatmap)
    h_norm = exp_heatmap / torch.sum(exp_heatmap, dim = (-1,-2), keepdim = True)
    
    #DISCRETE FORM of the Integral Equation
    # all locations p in the domain, 
    x_list = torch.linspace(0,1,h).cuda()#Why 0 and 1? it is easy to manipulate values btw 0 and 1
    y_list = torch.linspace(0,1,w).cuda()
    # 3D Heatmap z_list = torch.linspace(0,1,z).cuda()
    i,j = torch.meshgrid(x_list, y_list)

    #weighted by their probabilities.
    i_ = torch.sum(i*h_norm, dim=(-1,-2))
    j_ = torch.sum(j*h_norm, dim=(-1,-2))

    #Modified arrangement
    pose  = torch.stack((i_,j_),dim=2) #[[i,i,i,,],
                                       #[j,j,j,,,]]

    return ({'probabilitymap': h_norm, 'pose_2d': pose}) #(N,K, 2)

def integral_3d_innovate(heatmap_):
    print('3d Innovate being used')
    #heatmap i.e pred_keypoint_logits (Tensor): A tensor of shape (N, 72, S, S) / (N, K, H, W) 
    h, w = heatmap.shape[2], heatmap.shape[3]

    # H Heatmap, X,Y,Z location maps
    heatmap = heatmap_[:,0:18,:,:]
    location_map_X = heatmap_[:,18:36,:,:]
    location_map_Y = heatmap_[:,36:54,:,:]
    location_map_Z = heatmap_[:,54:72,:,:]

     #implementing softmax ,#soving the numerical problem
    heatmap = heatmap - torch.max(torch.max(heatmap, dim=-1)[0], dim=-1, keepdim=True)[0].unsqueeze(-1) 
    exp_heatmap = torch.exp(heatmap)
    h_norm = exp_heatmap / torch.sum(exp_heatmap, dim = (-1,-2), keepdim = True)
    
    #DISCRETE FORM of the Integral Equation
    # all locations p in the domain, 
    #x_list = torch.linspace(0,1,h).cuda()#Why 0 and 1? it is easy to manipulate values btw 0 and 1
    #y_list = torch.linspace(0,1,w).cuda()
    # 3D Heatmap z_list = torch.linspace(0,1,z).cuda()
    #i,j = torch.meshgrid(x_list, y_list)



    #weighted by their probabilities.
    i_ = torch.sum(location_map_X*h_norm, dim=(-1,-2))
    j_ = torch.sum(location_map_Y*h_norm, dim=(-1,-2))
    z_ = torch.sum(location_map_Z*h_norm, dim=(-1,-2))

    pose3d  = torch.stack((i_,j_,z_),dim=2) #[[i,i,i,,],
                                       #[j,j,j,,,]]

    return ({'probabilitymap': h_norm, 'pose_3d': pose3d}) #(N,K,3)

def keypoint_rcnn_loss(pred_keypoint_logits, instances, normalizer, use_2d = True):
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

    keypoint_side_len = pred_keypoint_logits.shape[2]
    #M = len(instances)
    #kps =  torch.zeros(M, )
    for instances_per_image in instances:
        if len(instances_per_image) == 0:
            continue
        keypoints = instances_per_image.gt_keypoints
        #e.g (8,6,3)
        print('Daniel test keypoints', keypoints.tensor.shape)
        #GT keypoints -> GT heatmaps  
        heatmaps_per_image, valid_per_image = keypoints.to_heatmap(
            instances_per_image.proposal_boxes.tensor, keypoint_side_len
        )
        #GT heatmaps -> to 1D vector
        heatmaps.append(heatmaps_per_image.view(-1)) #N*K
        valid.append(valid_per_image.view(-1)) #stretch to 1D vector
        #print('keypoints.tensor[:,:,0:2]', keypoints.tensor[:,:,0:2].shape)
        kps.append(keypoints.tensor[:,:,0:2]) #exclude visibility out

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

    N, K, H, W = pred_keypoint_logits.shape

    # pred_keypoint_logits = pred_keypoint_logits.view(N * K, H * W)
    # pred_keypoint_logits_  = pred_keypoint_logits[valid].view(N,K, H,W)
    #pred_keypoint_logits = pred_keypoint_logits.view(N * K, H * W)
    if use_2d:
        print('pred_keypoint_logits', pred_keypoint_logits[0][0:2])
        print('using 2d innovate')
        pred_integral = integral_2d_innovate(pred_keypoint_logits)
        #print('raw 2d pred integral output: ', pred_integral['pose_2d'].shape)
        pred_integral = pred_integral['pose_2d'].view(N * K, -1)[valid]

    else: #3d
        pred_integral = integral_3d_innovate(pred_keypoint_logits)
        print('raw 3d pred integral output: ', pred_integral['pose_3d'].shape)
        pred_integral = pred_integral['pose_3d'].view(N * K, -1)[valid]
    
    #print('pred_integral removed shape', pred_integral.shape)
    kps = torch.cat(kps)

    #normalize kps
    #All data mean-std normalization
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

    #With Batch mean and std, you wont have fixed values to de-normarlize

    #kps = (kps - kp_mean)/kp_std

    #Min-max Normalization]
    xmax, xmin, ymax, ymin = 1236.8367, 0.0, 619.60706, 8.637619

    partx = kps[:,:,0:1]
    partx = (partx - xmin)/(xmax - xmin)

    party = kps[:,:,1:2]
    party = (party - ymin)/(ymax - ymin)

    kps = torch.stack((partx,  party), dim = -2)
    #print('1st kps', kps.shape)
    kps = kps.squeeze(-1)


    #print('raw kps shape', kps.shape)
    #keypoint_loss = torch.nn.functional.mse_loss(pred_integral, keypoint_targets[valid])
    s1, s2 = kps.shape[0], kps.shape[1] #shape
    kps = kps.view(s1*s2, -1)[valid]
    #print('kps removed shape', kps.shape)


    print('pred: ', pred_integral[0:3], pred_integral[-3:])
    print('kps: ', kps[0:3], kps[-3:])
    print()
    print('final kps shape',kps.shape, 'final pred shape', pred_integral.shape)
    keypoint_loss = torch.nn.functional.mse_loss(pred_integral, kps)
    #print()
    print('raw loss', keypoint_loss)



    # keypoint_loss = F.cross_entropy(
    #     pred_keypoint_logits[valid], keypoint_targets[valid], reduction="sum"
    # )

    # If a normalizer isn't specified, normalize by the number of visible keypoints in the minibatch
    if normalizer is None:
        normalizer = valid.numel()
    #keypoint_loss /= normalizer

    #print('normalized loss: ', keypoint_loss, 'normalizer amount: ', normalizer)
    print()
    return keypoint_loss


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
    out = integral_2d_innovate(pred_keypoint_logits)
    heatmap_norm = out['probabilitymap']
    scores = torch.max(torch.max(heatmap_norm, dim = -1)[0], dim = -1)[0]
    #unstack
    i_, j_  = torch.unbind(out['pose_2d'], dim=2)

    #de-normalize
    xmax, xmin, ymax, ymin = 1236.8367, 0.0, 619.60706, 8.637619
    i_ = (i_ * (xmax - xmin)) + xmin
    j_ = (j_ * (ymax - ymin)) + ymin

    #instance, K, 3) 3-> (x, y, score)
    keypoint_results = torch.stack((i_,j_, scores),dim=2)
    print('keypoint_results', keypoint_results.shape)
    num_instances_per_image = [len(i) for i in pred_instances]
    keypoint_results = keypoint_results[:, :, [0, 1, 3]].split(num_instances_per_image, dim=0)
    print('pred_instances', len(pred_instances))

    for keypoint_results_per_image, instances_per_image in zip(keypoint_results, pred_instances):
        # keypoint_results_per_image is (num instances)x(num keypoints)x(x, y, score)
        
        print('=== keypoint_results_per_image:', keypoint_results_per_image.shape)
        #print('type:', instances_per_image.pred_keypoints.shape)
        instances_per_image.pred_keypoints = keypoint_results_per_image #.unsqueeze(0)
        
    #instances_per_image.pred_keypoints = keypoint_results


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

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = {
            "loss_weight": cfg.MODEL.ROI_KEYPOINT_HEAD.LOSS_WEIGHT,
            "num_keypoints": cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS,
        }
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
                "loss_keypoint": keypoint_rcnn_loss(x, instances, normalizer=normalizer)
                * self.loss_weight
            }
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
                # Caffe2 implementation uses MSRAFill, which in fact
                # corresponds to kaiming_normal_ in PyTorch
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")

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
