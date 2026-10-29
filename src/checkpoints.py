# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
from collections.abc import Mapping
from pathlib import Path

import torch


def load_weights(model, path, *, strict=True):
    checkpoint = torch.load(Path(path).expanduser(), map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("checkpoint must contain a tensor state dictionary")
    state_dict = checkpoint.get("state_dict", checkpoint)
    if (not isinstance(state_dict, Mapping) or not state_dict
            or not all(isinstance(key, str) and isinstance(value, torch.Tensor)
                       for key, value in state_dict.items())):
        raise ValueError("checkpoint state_dict must map names to tensors")
    return model.load_state_dict(state_dict, strict=strict)
