# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import cv2
import numpy as np

import shapely
from shapely.geometry import Polygon
import pygeoops

from torch.optim.lr_scheduler import _LRScheduler
from src.safety import MAX_CLASSES, bounded_integer, image_dimensions

cmap = np.array([[0,0,0], [255,255,255]])
MAX_CONTOURS = 128
MAX_CONTOUR_POINTS = 512
MAX_TOTAL_CONTOUR_POINTS = 4096


def colorize(labels, n_classes):
    bounded_integer(n_classes, "num_classes", 2, MAX_CLASSES)
    labels = np.asarray(labels)
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("class labels must be integers")
    valid = (labels >= 0) & (labels < n_classes)
    if np.any(~valid & (labels != 255) & (labels != -1)):
        raise ValueError("class labels are outside num_classes")
    palette = np.array([[(i * 37) % 256, (i * 67) % 256, (i * 97) % 256]
                        for i in range(n_classes)], dtype=np.uint8)
    palette[:2] = cmap
    colored = np.full(labels.shape + (3,), 128, dtype=np.uint8)
    colored[valid] = palette[labels[valid]]
    return colored


def softmax(x, T=1, axis=0):
    if not np.isfinite(T) or T <= 0 or not np.all(np.isfinite(x)):
        raise ValueError("softmax requires finite logits and a positive temperature")
    scaled = x / T
    e_x = np.exp(scaled - np.max(scaled, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)


class _StreamMetrics(object):
    def __init__(self):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def update(self, gt, pred):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def get_results(self):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def to_str(self, metrics):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def reset(self):
        """ Overridden by subclasses """
        raise NotImplementedError()      


class StreamSegMetrics(_StreamMetrics):
    """
    Stream Metrics for Semantic Segmentation Task
    """
    def __init__(self, n_classes):
        self.n_classes = bounded_integer(n_classes, "num_classes", 2, MAX_CLASSES)
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def update(self, label_trues, label_preds):
        label_trues, label_preds = np.asarray(label_trues), np.asarray(label_preds)
        if label_trues.shape != label_preds.shape:
            raise ValueError("label and prediction shapes must match")
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist( lt.flatten(), lp.flatten() )
    
    @staticmethod
    def to_str(results):
        string = "\n"
        for k, v in results.items():
            if k!="Class IoU":
                string += "%s: %f\n"%(k, v)
        
        #string+='Class IoU:\n'
        #for k, v in results['Class IoU'].items():
        #    string += "\tclass %d: %f\n"%(k, v)
        return string

    def _fast_hist(self, label_true, label_pred):
        if (label_true.shape != label_pred.shape
                or not np.issubdtype(label_true.dtype, np.integer)
                or not np.issubdtype(label_pred.dtype, np.integer)):
            raise ValueError("labels and predictions must have matching shapes and integer dtypes")
        if np.any((label_pred < 0) | (label_pred >= self.n_classes)):
            raise ValueError("predicted class is outside num_classes")
        mask = (label_true >= 0) & (label_true < self.n_classes)
        if np.any(~mask & (label_true != 255) & (label_true != -1)):
            raise ValueError("target class is outside num_classes")
        hist = np.bincount(
            self.n_classes * label_true[mask].astype(np.int64) + label_pred[mask].astype(np.int64),
            minlength=self.n_classes ** 2,
        ).reshape(self.n_classes, self.n_classes)
        return hist

    def get_results(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        if hist.sum() == 0:
            raise ValueError("cannot compute metrics without valid labeled pixels")
        acc = np.diag(hist).sum() / hist.sum()
        present = hist.sum(axis=1) > 0
        acc_per_cls = np.divide(np.diag(hist), hist.sum(axis=1),
                                out=np.zeros(self.n_classes), where=present)
        acc_cls = acc_per_cls[present].mean()
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist) + 1e-5)
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))

        return {
                "Overall Acc": acc,
                "Mean Acc": acc_cls,
                "Per Cls Acc": acc_per_cls,
                "FreqW Acc": fwavacc,
                "Mean IoU": mean_iu,
                "Class IoU": cls_iu,
            }
        
    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


class AverageMeter(object):
    """Computes average values"""
    def __init__(self):
        self.book = dict()

    def reset_all(self):
        self.book.clear()
    
    def reset(self, id):
        item = self.book.get(id, None)
        if item is not None:
            item[0] = 0
            item[1] = 0

    def update(self, id, val):
        record = self.book.get(id, None)
        if record is None:
            self.book[id] = [val, 1]
        else:
            record[0]+=val
            record[1]+=1

    def get_results(self, id):
        record = self.book.get(id, None)
        assert record is not None
        return record[0] / record[1]

        

class PolyLR(_LRScheduler):
    def __init__(self, optimizer, max_iters, power=0.9, last_epoch=-1, min_lr=1e-6):
        self.power = power
        self.max_iters = max_iters  # avoid zero lr
        self.min_lr = min_lr
        super(PolyLR, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        return [ max( base_lr * ( 1 - self.last_epoch/self.max_iters )**self.power, self.min_lr)
                for base_lr in self.base_lrs]

                
def denormalize(data, mean, std):
    mean = np.array(mean)
    std = np.array(std)

    _mean = -mean/std
    _std = 1/std

    return (data -_mean.reshape((3, 1, 1))) / _std.reshape((3,1,1)) 


def prediction_outputs(pred, probs, default_size=(2000, 500), area_threshold=100):
    pred, probs = np.asarray(pred), np.asarray(probs)
    if pred.ndim != 2 or pred.shape != probs.shape:
        raise ValueError("prediction and probability maps must have the same 2D shape")
    image_dimensions(pred.shape[1], pred.shape[0])
    image_dimensions(*default_size)
    if (not np.all(np.isin(pred, [0, 1])) or not np.all(np.isfinite(probs))
            or np.any((probs < 0) | (probs > 1))):
        raise ValueError("geometry extraction requires binary predictions and finite probabilities")
    if not np.isfinite(area_threshold) or area_threshold < 0:
        raise ValueError("area_threshold must be finite and nonnegative")
    polygon_coords = []
    centroids = []
    center_lines = []
    mean_conf = []
    med_conf = []

    contours_pred, _ = cv2.findContours((pred * 225).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if (len(contours_pred) > MAX_CONTOURS
            or sum(len(contour) for contour in contours_pred) > MAX_TOTAL_CONTOUR_POINTS
            or any(len(contour) > MAX_CONTOUR_POINTS for contour in contours_pred)):
        raise ValueError("prediction geometry exceeds contour or vertex limits")

    for i in range(len(contours_pred)):

        contour = contours_pred[i]

        if len(contour) > 10: # we should at least have 10 points in a polygon

            polygon = Polygon(contour.reshape(-1, 2))

            if polygon.area > area_threshold:
                if not polygon.is_valid:
                    raise ValueError("prediction contains an invalid polygon")
                
                # First, polygon coords
                coords = []
                for coord in contour.reshape(-1, 2):
                    coords.append([coord[0].item() / default_size[0],
                                   coord[1].item() / default_size[1]])

                polygon_coords.append(coords)

                # Second, centroids and center lines
                # Disable automatic densification so the vertex budget also bounds Voronoi input.
                line_geom = pygeoops.centerline(polygon, densify_distance=0)
                if (line_geom is None or line_geom.is_empty
                        or not isinstance(line_geom, (shapely.geometry.LineString,
                                                      shapely.geometry.MultiLineString))):
                    raise ValueError("could not derive a centerline from the prediction polygon")
                length = line_geom.length
                cent = line_geom.interpolate(length/2.0).coords[0]
                centroids.append((cent[0] / default_size[0],
                                  cent[1] / default_size[1]))

                c_lines = []
                if type(line_geom) is shapely.geometry.linestring.LineString:
                    c_lines.append([[c[0] / default_size[0],
                                      c[1] / default_size[1]]
                                    for c in line_geom.coords])
                elif type(line_geom) is shapely.geometry.multilinestring.MultiLineString:
                    for geom in line_geom.geoms:
                        c_lines.append([[c[0] / default_size[0],
                                         c[1] / default_size[1]]
                                        for c in geom.coords])
                center_lines.append(c_lines)

                # Third, mean and median confidence
                cimg = np.zeros_like(probs, dtype=np.uint8)
                cv2.drawContours(cimg, [contour], -1, color=1, thickness=-1)
                
                mean_conf.append(np.mean(probs[cimg == 1]).item())
                med_conf.append(np.median(probs[cimg == 1]).item())
    
    return polygon_coords, centroids, center_lines, mean_conf, med_conf