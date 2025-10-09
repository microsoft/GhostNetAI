# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import time
import os
import numpy as np
import json
from PIL import Image

import torch
import torch.optim as optim
import lightning as L

from src import models
from .utils import StreamSegMetrics, PolyLR, denormalize, cmap, prediction_outputs, softmax

__all__ = ['Plain']

class Plain(L.LightningModule):

    """
    Overall training function.
    """

    name = 'Plain'

    def __init__(self, conf, **kwargs):
        super().__init__()
        self.hparams.update(conf.__dict__)
        self.save_hyperparameters(ignore=['conf', 'train_class_counts'])
        self.net = models.__dict__[self.hparams.model_name](num_cls=self.hparams.num_classes)
        self.metrics = StreamSegMetrics(self.hparams.num_classes)

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

    def validation_step(self, batch, batch_idx):
        data, labels, _ = batch
        outputs = self.net(data)
        preds = outputs.detach().max(dim=1)[1]
        self.metrics.update(labels.cpu().numpy(), preds.cpu().numpy())

        self.val_st_outs.append((data.detach().cpu().numpy(),
                                 preds.detach().cpu().numpy(),
                                 labels.cpu().numpy()))

    def on_validation_epoch_end(self):
        scores = self.metrics.get_results()
        self.log('valid_mean_IoU', scores['Mean IoU'])
        self.log('valid_overall_acc', scores['Overall Acc'])
        self.log('valid_mean_acc', scores['Mean Acc'])
        for i in range(len(scores['Per Cls Acc'])):
            self.log('cls_{}_acc'.format(i), scores['Per Cls Acc'][i])

        total_data = np.concatenate([x[0] for x in self.val_st_outs], axis=0)
        total_preds = np.concatenate([x[1] for x in self.val_st_outs], axis=0)
        total_labels = np.concatenate([x[2] for x in self.val_st_outs], axis=0)

        # for i in np.random.choice(len(total_labels), 10, replace=False):
        # np.random.seed(10)
        for i in range(len(total_labels)):
            image = total_data[i]
            target = total_labels[i]
            pred = total_preds[i]

            image = (denormalize(image, 
                                 mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225]) * 255).transpose(1, 2, 0).astype(np.uint8)
            target = cmap[target].astype(np.uint8)
            pred = cmap[pred].astype(np.uint8)

            image = Image.fromarray(image)
            target = Image.fromarray(target)
            pred = Image.fromarray(pred)

            if self.logger.__class__.__name__ == 'TensorboardLogger':
                self.logger.experiment.log_image(image, name='{}_data.png'.format(i), overwrite=True)
                self.logger.experiment.log_image(target, name='{}_target.png'.format(i), overwrite=True)
                self.logger.experiment.log_image(pred, name='{}_pred.png'.format(i), overwrite=True)
            else:
                os.makedirs('./log/temp_imgs', exist_ok=True)
                image.save('./log/temp_imgs/{}_data.png'.format(i))
                target.save('./log/temp_imgs/{}_target.png'.format(i))
                pred.save('./log/temp_imgs/{}_pred.png'.format(i))

    def on_test_start(self):
        self.metrics.reset()
        self.te_st_outs = []

    def test_step(self, batch, batch_idx):
        data, labels, file_ids = batch
        outputs = self.net(data)
        preds = outputs.detach().max(dim=1)[1]
        self.metrics.update(labels.cpu().numpy(), preds.cpu().numpy())
        self.te_st_outs.append((data.detach().cpu().numpy(), 
                                preds.cpu().numpy(),
                                labels.cpu().numpy(),
                                outputs.cpu().numpy(),
                                file_ids))

    def on_test_epoch_end(self):
        scores = self.metrics.get_results()
        self.log('valid_mean_IoU', scores['Mean IoU'])
        self.log('valid_overall_acc', scores['Overall Acc'])
        self.log('valid_mean_acc', scores['Mean Acc'])
        for i in range(len(scores['Per Cls Acc'])):
            self.log('cls_{}_acc'.format(i), scores['Per Cls Acc'][i])

        total_data = np.concatenate([x[0] for x in self.te_st_outs], axis=0)
        total_preds = np.concatenate([x[1] for x in self.te_st_outs], axis=0)
        total_labels = np.concatenate([x[2] for x in self.te_st_outs], axis=0)
        total_outputs = np.concatenate([x[3] for x in self.te_st_outs], axis=0)
        total_file_ids = np.concatenate([x[4] for x in self.te_st_outs], axis=0)

        np.savez(self.hparams.evaluate.replace('.ckpt', '.npz'), 
                 data=total_data, preds=total_preds, targets=total_labels,
                 logits=total_outputs, file_ids=total_file_ids)
        
        print('Evaluation outputs saved to {}.'.format(self.hparams.evaluate.replace('.ckpt', '.npz')))
    
    def on_predict_start(self):
        self.json_outputs = []

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
            self.json_outputs.append({
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
        output_path_json = self.hparams.evaluate.replace('.ckpt', '_predict.json') 
        json.dump(self.json_outputs, open(output_path_json, 'w'))
        print('Predict output json saved to {}.'.format(output_path_json))