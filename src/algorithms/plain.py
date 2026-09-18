# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import os
from pathlib import Path
import numpy as np
from PIL import Image

import torch
import torch.optim as optim
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger

from src import models
from src.config import select_component, validate_config
from .output import JsonArrayWriter, NpzWriter
from .utils import StreamSegMetrics, PolyLR, denormalize, colorize, prediction_outputs, softmax

__all__ = ['Plain']
MAX_VALIDATION_SAMPLES = 10
MAX_VALIDATION_BYTES = 128 * 1024 * 1024

class Plain(L.LightningModule):

    """
    Overall training function.
    """

    name = 'Plain'

    def __init__(self, conf, **kwargs):
        super().__init__()
        validate_config(conf)
        self.hparams.update(conf.__dict__)
        self.save_hyperparameters(ignore=['conf', 'train_class_counts'])
        self.metrics = StreamSegMetrics(self.hparams.num_classes)
        model_class = select_component(models.MODELS, self.hparams.model_name, "model")
        self.net = model_class(num_cls=self.hparams.num_classes)

    def configure_optimizers(self):
        net_optim_params_list = [
            {'params': self.net.feature.parameters(),
            #  'lr': self.hparams.lr_feature*torch.cuda.device_count(),
             'lr': self.hparams.lr_feature,
             'momentum': self.hparams.momentum_feature,
             'weight_decay': self.hparams.weight_decay_feature},
            {'params': self.net.classifier.parameters(),
            #  'lr': self.hparams.lr_classifier*torch.cuda.device_count(),
             'lr': self.hparams.lr_classifier,
             'momentum': self.hparams.momentum_classifier,
             'weight_decay': self.hparams.weight_decay_classifier}
        ]
        # Setup optimizer and optimizer scheduler
        optimizer = torch.optim.SGD(net_optim_params_list)
        scheduler = PolyLR(optimizer, self.hparams.num_iters, power=0.9)
        return [optimizer], [{'scheduler': scheduler, 'interval': 'step', 'frequency': 1}]

    def on_train_start(self):
        self.net.setup_criteria()

    def training_step(self, batch, batch_idx):
        data, labels, _ = batch
        outputs = self.net(data)
        loss = self.net.criterion_seg(outputs, labels.long())
        self.log("train_loss", loss)
        return loss

    def on_validation_start(self):
        self.metrics.reset()
        self.val_st_outs = []
        self.validation_bytes = 0

    def validation_step(self, batch, batch_idx):
        data, labels, _ = batch
        outputs = self.net(data)
        preds = outputs.detach().max(dim=1)[1]
        self.metrics.update(labels.cpu().numpy(), preds.cpu().numpy())

        for image, pred, label in zip(data, preds, labels):
            if len(self.val_st_outs) >= MAX_VALIDATION_SAMPLES:
                break
            size = sum(tensor.numel() * tensor.element_size() for tensor in (image, pred, label))
            if self.validation_bytes + size > MAX_VALIDATION_BYTES:
                break
            self.val_st_outs.append(tuple(tensor.detach().cpu().numpy().copy()
                                          for tensor in (image, pred, label)))
            self.validation_bytes += size

    def on_validation_epoch_end(self):
        scores = self.metrics.get_results()
        self.log('valid_mean_IoU', scores['Mean IoU'])
        self.log('valid_overall_acc', scores['Overall Acc'])
        self.log('valid_mean_acc', scores['Mean Acc'])
        for i in range(len(scores['Per Cls Acc'])):
            self.log('cls_{}_acc'.format(i), scores['Per Cls Acc'][i])

        for i, (image, pred, target) in enumerate(self.val_st_outs):
            if not self.trainer.is_global_zero:
                break

            image = (denormalize(image, 
                                 mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225]) * 255).transpose(1, 2, 0).astype(np.uint8)
            target = colorize(target, self.hparams.num_classes)
            pred = colorize(pred, self.hparams.num_classes)

            image = Image.fromarray(image)
            target = Image.fromarray(target)
            pred = Image.fromarray(pred)

            if isinstance(self.logger, TensorBoardLogger):
                for name, value in (("data", image), ("target", target), ("pred", pred)):
                    self.logger.experiment.add_image(
                        f"validation/{i}_{name}", np.asarray(value),
                        global_step=self.global_step, dataformats="HWC",
                    )
            else:
                os.makedirs('./log/temp_imgs', exist_ok=True)
                image.save('./log/temp_imgs/{}_data.png'.format(i))
                target.save('./log/temp_imgs/{}_target.png'.format(i))
                pred.save('./log/temp_imgs/{}_pred.png'.format(i))
        self.val_st_outs.clear()
        self.validation_bytes = 0

    def _output_path(self, suffix):
        path = Path(self.hparams.evaluate).expanduser()
        rank = f"_rank{self.global_rank}" if self.trainer.world_size > 1 else ""
        return path.with_name(path.stem + rank + suffix)

    def on_test_start(self):
        self.metrics.reset()
        self.test_writer = NpzWriter(self._output_path(".npz"))

    def test_step(self, batch, batch_idx):
        data, labels, file_ids = batch
        outputs = self.net(data)
        preds = outputs.detach().max(dim=1)[1]
        self.metrics.update(labels.cpu().numpy(), preds.cpu().numpy())
        self.test_writer.append(
            data=data.detach().cpu().numpy(), preds=preds.cpu().numpy(),
            targets=labels.cpu().numpy(), logits=outputs.detach().cpu().numpy(),
            file_ids=np.asarray(file_ids, dtype=str),
        )

    def on_test_epoch_end(self):
        scores = self.metrics.get_results()
        self.log('valid_mean_IoU', scores['Mean IoU'])
        self.log('valid_overall_acc', scores['Overall Acc'])
        self.log('valid_mean_acc', scores['Mean Acc'])
        for i in range(len(scores['Per Cls Acc'])):
            self.log('cls_{}_acc'.format(i), scores['Per Cls Acc'][i])

        self.test_writer.finalize()
        print('Evaluation outputs saved to {}.'.format(self.test_writer.path))
    
    def on_predict_start(self):
        if self.hparams.num_classes != 2:
            raise ValueError("polygon prediction requires exactly two classes")
        self.prediction_writer = JsonArrayWriter(self._output_path("_predict.json"))

    def predict_step(self, batch, batch_idx):
        data, file_ids, ori_sizes = batch
        ori_sizes = ori_sizes.cpu().numpy()
        outputs = self.net(data)
        preds = outputs.detach().max(dim=1)[1].cpu().numpy()
        outputs = outputs.detach().cpu().numpy()

        # probs = softmax(outputs.detach().cpu().numpy(), axis=1, T=1.1)

        for i in range(len(preds)):
            pred = preds[i]
            prob = softmax(outputs[i], axis=0, T=1.1)
            polygons, centroids, center_lines, mean_conf, med_conf = prediction_outputs(pred.copy(), prob[1].copy(), default_size=(2000, 500))
            self.prediction_writer.append({
                "marker_id": "",
                "survey_pic_id": file_ids[i],
                "original_image_size": ori_sizes[i].tolist(),
                "marker_confidence": "",
                "marker_gear_type": ["ghostnet" for _ in range(len(polygons))],
                "marker_bounding_polygon": polygons,
                "marker_centroid": centroids,
                "marker_center_lines": center_lines,
                "polygon_mean_confidence": mean_conf,
                "polygon_med_confidence": med_conf,
                "marker_status": "unverified",
                "marker_ai_model": self.hparams.model_name,
            })
        
    def on_predict_epoch_end(self):
        self.prediction_writer.finalize()
        print('Predict output json saved to {}.'.format(self.prediction_writer.path))

    def teardown(self, stage):
        for name in ("test_writer", "prediction_writer"):
            writer = getattr(self, name, None)
            if writer is not None:
                writer.abort()