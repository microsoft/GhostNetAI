# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
from .deeplab_v3 import DeepLabV3_ResNet50

MODELS = {"DeepLabV3_ResNet50": DeepLabV3_ResNet50}
__all__ = ["DeepLabV3_ResNet50"]
