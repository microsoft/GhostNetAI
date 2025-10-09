# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models.resnet import resnet50
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision.models._utils import IntermediateLayerGetter

__all__ = ['DeepLabV3_ResNet50']

class DeepLabV3_ResNet50(nn.Module):

    name = 'DeepLabV3_ResNet50'

    def __init__(self, num_cls=21):
        super(DeepLabV3_ResNet50, self).__init__()
        self.num_cls = num_cls
        self.feature = None
        self.classifier = None
        self.criterion_seg = None
        self.setup_net()

    def setup_net(self):
        self.feature = IntermediateLayerGetter(resnet50(pretrained=True, 
                                                        replace_stride_with_dilation=[False, True, True]),
                                               return_layers = {'layer4': 'out'}) 
        self.classifier = DeepLabHead(2048, self.num_cls)

        for m in self.feature.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.momentum = 0.01

    def setup_criteria(self):
        self.criterion_seg = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')

    def forward(self, data):
        input_shape = data.shape[-2:]
        features = self.feature(data)
        outputs = self.classifier(features['out'])
        # outputs = self.classifier(features)
        outputs = F.interpolate(outputs, size=input_shape, mode='bilinear', align_corners=False)
        return outputs

