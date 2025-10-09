# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# %%
import numpy as np
from PIL import Image, ImageDraw
from matplotlib import cm
import cv2

import shapely
from shapely.geometry import Polygon
import pygeoops


# %%
cmap = np.array([[0,0,0], [255,255,255]])

def denormalize(data, mean, std):
    mean = np.array(mean)
    std = np.array(std)
    _mean = -mean/std
    _std = 1/std
    return (data -_mean.reshape((3, 1, 1))) / _std.reshape((3,1,1)) 

def softmax(x, T=1, axis=0):
    e_x = np.exp(x / T)
    return e_x / e_x.sum(axis=axis)


# %%
outputs_np = np.load('../weights/GhostSeg_DE/Plain/GhostSeg_DE_Plain_111824-0-epoch=2514-valid_mean_IoU=0.77.npz')

total_data = outputs_np['data']
total_preds = outputs_np['preds']
total_labels = outputs_np['targets']
total_logits = outputs_np['logits']
total_file_ids = outputs_np['file_ids']


# %%
idx = 11
cat = 1
data, target, pred, logits, file_id = total_data[idx], total_labels[idx], total_preds[idx], total_logits[idx], total_file_ids[idx]
probs = softmax(logits, axis=0, T=1.1)

image = (denormalize(data, 
                     mean=[0.485, 0.456, 0.406], 
                     std=[0.229, 0.224, 0.225]) * 255).transpose(1, 2, 0).astype(np.uint8)
target_img = (target * 225).astype(np.uint8)
pred_img = (pred * 225).astype(np.uint8)

image = Image.fromarray(image)
target_img = Image.fromarray(target_img)
pred_img = Image.fromarray(pred_img)
probs_img = Image.fromarray(np.uint8(cm.coolwarm(probs[cat]) * 255)).convert('RGB')


# %%
pred_cont = np.array(image.copy())
target_pred_cont = np.array(image.copy())

contours_pred, _ = cv2.findContours((pred * 225).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours_target, _ = cv2.findContours((target * 225).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(pred_cont, contours_pred, -1, (255, 0, 0), 3)

cv2.drawContours(target_pred_cont, contours_pred, -1, (255, 0, 0), 3)
cv2.drawContours(target_pred_cont, contours_target, -1, (0, 0, 255), 3)

pred_overlay = Image.fromarray(pred_cont)
target_pred_overlay = Image.fromarray(target_pred_cont)

conf_overlay = cv2.addWeighted(np.array(probs_img.copy()), 0.3,
                               np.array(image.copy()), 0.7, 0)
conf_overlay = Image.fromarray(conf_overlay)


# %%
pred_overlay

# %%
target_pred_overlay

# %%
conf_overlay


# %%
# Generate mid points
pred_cent_overlay = target_pred_overlay.copy()
draw = ImageDraw.Draw(pred_cent_overlay)

for i in range(len(contours_pred)):
# for i in range(1, 2):
    try:
        if len(contours_pred[i]) > 10:
            polygon = Polygon(contours_pred[i].reshape(-1, 2))
            if polygon.area > 100:
                line_geom = pygeoops.centerline(polygon)
                length = line_geom.length
                centroid = line_geom.interpolate(length/2.0).coords[0] #QgsGeometry representing the mid point

                draw.ellipse((centroid[0]-7, centroid[1]-7, centroid[0]+7, centroid[1]+7),
                             fill = 'Yellow', outline ='Yellow')

    except:
        print(i)

# %%
pred_cent_overlay

# %%
