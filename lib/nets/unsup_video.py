from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from nets.network import Network
from model.config import cfg

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torchvision.models as models

import torch
import torch.nn as nn
import torch.legacy.nn as lnn

from functools import reduce
from torch.autograd import Variable

class LambdaBase(nn.Sequential):
    def __init__(self, fn, *args):
        super(LambdaBase, self).__init__(*args)
        self.lambda_func = fn

    def forward_prepare(self, input):
        output = []
        for module in self._modules.values():
            output.append(module(input))
        return output if output else input

class Lambda(LambdaBase):
    def forward(self, input):
        return self.lambda_func(self.forward_prepare(input))

class LambdaMap(LambdaBase):
    def forward(self, input):
        return list(map(self.lambda_func,self.forward_prepare(input)))

class LambdaReduce(LambdaBase):
    def forward(self, input):
        return reduce(self.lambda_func,self.forward_prepare(input))

# unsup_video = nn.Sequential( # Sequential,
# 	nn.Conv2d(3,96,(11, 11),(4, 4)),
# 	nn.ReLU(),
# 	nn.MaxPool2d((3, 3),(2, 2)),
# 	Lambda(lambda x,lrn=lnn.SpatialCrossMapLRN(*(5, 0.0001, 0.75, 1)): Variable(lrn.forward(x.data))),
# 	nn.Conv2d(96,256,(5, 5),(1, 1),(2, 2)),
# 	nn.ReLU(),
# 	nn.MaxPool2d((3, 3),(2, 2)),
# 	Lambda(lambda x,lrn=lnn.SpatialCrossMapLRN(*(5, 0.0001, 0.75, 1)): Variable(lrn.forward(x.data))),
# 	nn.Conv2d(256,384,(3, 3),(1, 1),(1, 1)),
# 	nn.ReLU(),
# 	nn.Conv2d(384,384,(3, 3),(1, 1),(1, 1)),
# 	nn.ReLU(),
# 	nn.Conv2d(384,256,(3, 3),(1, 1),(1, 1)),
# 	nn.ReLU(),
# 	nn.MaxPool2d((3, 3),(2, 2)),
# 	nn.Sequential( # Sequential,
# 		Lambda(lambda x: x.view(x.size(0),-1)), # View,
# 		nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),nn.Linear(9216,4096)), # Linear,
# 	),
# 	nn.ReLU(),
# 	nn.Dropout(0.5),
# 	nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),nn.Linear(4096,4096)), # Linear,
# 	nn.ReLU(),
# 	nn.Dropout(0.5),
# )

class UnsupFeature(nn.Module):
  def __init__(self, in_channels=3):
    super(UnsupFeature, self).__init__()
    self.features = nn.Sequential( # Sequential,
        nn.Conv2d(in_channels,96,(11, 11),(4, 4)),
        nn.ReLU(),
        nn.MaxPool2d((3, 3),(2, 2)),
        Lambda(lambda x,lrn=lnn.SpatialCrossMapLRN(*(5, 0.0001, 0.75, 1)): Variable(lrn.forward(x.data))),
        nn.Conv2d(96,256,(5, 5),(1, 1),(2, 2)),
        nn.ReLU(),
        nn.MaxPool2d((3, 3),(2, 2)),
        Lambda(lambda x,lrn=lnn.SpatialCrossMapLRN(*(5, 0.0001, 0.75, 1)): Variable(lrn.forward(x.data))),
        nn.Conv2d(256,384,(3, 3),(1, 1),(1, 1)),
        nn.ReLU(),
        nn.Conv2d(384,384,(3, 3),(1, 1),(1, 1)),
        nn.ReLU(),
        nn.Conv2d(384,256,(3, 3),(1, 1),(1, 1)),
        nn.ReLU(),
        nn.MaxPool2d((3, 3),(2, 2)),
      )
    self.classifier = nn.Sequential(
        nn.Linear(9216,4096),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(4096,4096), # Linear,
        nn.ReLU(),
        nn.Dropout(0.5),
    )

  def forward(self, x):
    h = self.features(x)
    x = self.classifier(h.view(-1, 4096))
    return x

class unsup_video(Network):
  def __init__(self):
        Network.__init__(self)
        self._feat_stride = [
            16,
        ]
        self._feat_compress = [
            1. / float(self._feat_stride[0]),
        ]
        self._net_conv_channels = 256
        self._fc7_channels = 4096
        # self._fixed_layer = fixed_layer_num

  def _init_modules(self):
    self.unsup_video = UnsupFeature()
    # if self.pretrained:
    #     print("Loading pretrained weights from %s" %(self.model_path))
    #     state_dict = torch.load(self.model_path)
    #     unsup_video.load_state_dict({k:v for k,v in state_dict.items() if k in unsup_video.state_dict()})

    self.unsup_video.classifier = nn.Sequential(*list(unsup_video.classifier._modules.values()))

    for layer in range(13):
            for p in self.unsup_video.features.net[layer].parameters():
                p.requires_grad = False

    # not using the last maxpool layer
    self._layers['head'] = nn.Sequential(
            *list(self.unsup_video.features.net._modules.values())[:-1])

    def _image_to_head(self):
        net_conv = self._layers['head'](self._image)
        self._act_summaries['conv'] = net_conv

        return net_conv

    def _head_to_tail(self, pool5):
        pool5_flat = pool5.view(pool5.size(0), -1)
        fc7 = self.unsup_video.classifier(pool5_flat)

        return fc7

    def load_pretrained_cnn(self, state_dict):
        normal_dict = {}
        feature_dict = {}
        classifier_dict = {}
        isNormal = True
        for k, v in state_dict.items():
            if k.startswith('features'):
                if k in self.unsup_video.state_dict():
                    normal_dict[k] = v
            else:
                if k in self.unsup_video.features.state_dict():
                    isNormal = False
                    feature_dict[k] = v
                elif int(k[:2]) >= 15:
                    isNormal = False
                    classifier_dict[str(int(k[:2]) - 15) + k[k.rfind('.'):]] = v
        if isNormal:
            self.unsup_video.load_state_dict(normal_dict)
        else:
            self.unsup_video.features.load_state_dict(feature_dict)
            self.unsup_video.classifier.load_state_dict(classifier_dict)