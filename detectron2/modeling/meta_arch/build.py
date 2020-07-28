# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
from torch import nn

from detectron2.utils.registry import Registry

META_ARCH_REGISTRY = Registry("META_ARCH")  # noqa F401 isort:skip
META_ARCH_REGISTRY.__doc__ = """
Registry for meta-architectures, i.e. the whole model.

The registered object will be called with `obj(cfg)`
and expected to return a `nn.Module` object.
"""

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal(m.weight)

        

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

#Combine 2 architectures together
class CombinedWithLinear(nn.Module):
	"""docstring for Combined"""
	def __init__(self, maskrcc_model):
		super(CombinedWithLinear, self).__init__()
		self.maskrcc_model = maskrcc_model
		self.linearmodel = LinearModel()

	def foward(*input, **kwarg):
		print('################################')
		x = maskrcc_model(*input, **kwarg)
		print('================================')
		print('type:', type(x))
		print(x)
		x = linearmodel(x)
		return x


def build_model(cfg):
    """
    Build the whole model architecture, defined by ``cfg.MODEL.META_ARCHITECTURE``.
    Note that it does not load any weights from ``cfg``.
    """
    meta_arch = cfg.MODEL.META_ARCHITECTURE
    maskrcc_model = META_ARCH_REGISTRY.get(meta_arch)(cfg)
    print('maskrcc_model type', type(maskrcc_model))
    print('isinstance:', isinstance(maskrcc_model, nn.Module))
    
    model = CombinedWithLinear(maskrcc_model)
    model.to(torch.device(cfg.MODEL.DEVICE))

    return model

# def build_model(cfg):
#     """
#     Build the whole model architecture, defined by ``cfg.MODEL.META_ARCHITECTURE``.
#     Note that it does not load any weights from ``cfg``.
#     """
#     meta_arch = cfg.MODEL.META_ARCHITECTURE
#     model = META_ARCH_REGISTRY.get(meta_arch)(cfg)
#     model.to(torch.device(cfg.MODEL.DEVICE))
#     return model
