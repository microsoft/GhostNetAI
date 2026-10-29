# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import numpy as np
import torch

from src.algorithms.plain import Plain

__all__ = ['LA']

class LA(Plain):

    """
    Overall training function.
    """

    name = 'Logit adjustment'

    def __init__(self, conf, **kwargs):
        super(LA, self).__init__(conf=conf, **kwargs)
        label_freq_norm = np.array([0.97, 0.03]) 
        self.register_buffer(
            "adjustments",
            torch.from_numpy(np.log(label_freq_norm ** self.hparams.tau + 1e-12)).float(),
            persistent=False,
        )

    def training_step(self, batch, batch_idx):
        data, labels, _ = batch
        outputs = self.net(data)

        outputs = outputs + self.adjustments.to(outputs).reshape((1, 2, 1, 1))

        loss = self.net.criterion_seg(outputs, labels.long())

        loss_r = 0
        for parameter in self.net.parameters():
            loss_r += torch.sum(parameter ** 2)
        
        loss = loss + self.hparams.weight_decay_feature * loss_r
        if not torch.isfinite(loss):
            raise ValueError("training loss is non-finite")

        self.log("train_loss", loss)
        return loss