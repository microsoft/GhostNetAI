# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
from .datasets import GhostSeg_DE, GhostSeg_PS, GhostSeg_DEPS

DATASETS = {
    "GhostSeg_DE": GhostSeg_DE,
    "GhostSeg_PS": GhostSeg_PS,
    "GhostSeg_DEPS": GhostSeg_DEPS,
}
__all__ = ["GhostSeg_DE", "GhostSeg_PS", "GhostSeg_DEPS"]
