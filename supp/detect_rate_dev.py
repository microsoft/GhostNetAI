# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# %%    
import numpy as np
from PIL import Image
from matplotlib import cm
import cv2

import shapely
from shapely.geometry import Polygon
import pygeoops

# %%
def denormalize(data, mean, std):
    mean = np.array(mean)
    std = np.array(std)
    _mean = -mean/std
    _std = 1/std
    return (data -_mean.reshape((3, 1, 1))) / _std.reshape((3,1,1)) 

def softmax(x, T=1, axis=0):
    e_x = np.exp(x / T)
    return e_x / e_x.sum(axis=axis)

def prediction_outputs(pred, probs, default_size=(2000, 500), area_threshold=100):
    
    polygon_coords = []
    centroids = []
    center_lines = []
    mean_conf = []
    med_conf = []

    contours_pred, _ = cv2.findContours((pred * 225).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for i in range(len(contours_pred)):

        contour = contours_pred[i]

        if len(contour) > 10: # we should at least have 10 points in a polygon

            polygon = Polygon(contour.reshape(-1, 2))

            if polygon.area > area_threshold:
                
                # First, polygon coords
                coords = []
                for coord in contour.reshape(-1, 2):
                    coords.append([coord[0].item() / default_size[0],
                                   coord[1].item() / default_size[1]])

                polygon_coords.append(coords)

                # Second, centroids and center lines
                line_geom = pygeoops.centerline(polygon)
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

# %%
outputs_np = np.load('../weights/GhostSeg_PS/Plain/GhostSeg_PS_Plain_FT_001_merge_050425-0-epoch=658-valid_mean_IoU=0.72.npz')

total_data = outputs_np['data']
total_preds = outputs_np['preds']
total_labels = outputs_np['targets']
total_logits = outputs_np['logits']
total_file_ids = outputs_np['file_ids']

# %%
def evaluate_polygon_detection_centroid_with_buffer(pred_polygons, gt_polygons, buffer_distance=10.0):
    """
    Evaluate polygon detection using centroid-based matching with a buffer around ground truth polygons.
    
    Args:
        pred_polygons: List of predicted polygons (Shapely Polygon objects or coordinates).
        gt_polygons: List of ground truth polygons (Shapely Polygon objects or coordinates).
        buffer_distance: Distance to buffer ground truth polygons (default: 10.0 units).
    
    Returns:
        dict: Metrics including precision, recall, and F1-score.
    """
    # Convert coordinates to Shapely Polygons if necessary
    pred_polygons = [Polygon(p) if not isinstance(p, Polygon) else p for p in pred_polygons]
    gt_polygons = [Polygon(p) if not isinstance(p, Polygon) else p for p in gt_polygons]
    
    # Initialize counters
    tp, fp, fn = 0, 0, len(gt_polygons)
    matched_gt = set()
    
    # Check if predicted polygon's centroid lies within buffered ground truth polygon
    for pred_idx, pred_poly in enumerate(pred_polygons):
        pred_centroid = pred_poly.centroid
        matched = False
        
        for gt_idx, gt_poly in enumerate(gt_polygons):
            if gt_idx in matched_gt:
                continue
            # Create a buffered polygon around the ground truth polygon
            buffered_gt_poly = gt_poly.buffer(buffer_distance, resolution=16)
            if buffered_gt_poly.contains(pred_centroid):
                tp += 1
                fn -= 1
                matched_gt.add(gt_idx)
                matched = True
                break
        
        if not matched:
            fp += 1
    
    # Compute metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "buffer_distance": buffer_distance
    }

# %%
pred_polygons_all = []
target_polygons_all = []

for idx in range(len(total_data)):
    data, target, pred, logits, file_id = total_data[idx], total_labels[idx], total_preds[idx], total_logits[idx], total_file_ids[idx]
    prob = softmax(logits, axis=0, T=1.1)
    h, w = pred.shape

    polygons, centroids, center_lines, mean_conf, med_conf = prediction_outputs(pred.copy(), prob[1].copy(), default_size=(2000, 500))
    pred_polygons = [Polygon(np.array(p) * np.array([w, h])) for p in polygons]
    pred_polygons = [p for p in pred_polygons if p.area > 300]
    pred_polygons_all.extend(pred_polygons)
    # areas = [Polygon(np.array(p) * np.array([w, h])).area for p in polygons]

    contours_target, _ = cv2.findContours((target * 225).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    target_polygons = [Polygon(c.reshape(-1, 2)) for c in contours_target]
    target_polygons_all.extend(target_polygons)

# %%
metrics = evaluate_polygon_detection_centroid_with_buffer(pred_polygons_all, target_polygons_all, buffer_distance=5.0)
print("Evaluation Metrics:")
for key, value in metrics.items():
    print(f"{key}: {value}")


# %%
idx = 4
cat = 1

# %%
data, target, pred, logits, file_id = total_data[idx], total_labels[idx], total_preds[idx], total_logits[idx], total_file_ids[idx]
prob = softmax(logits, axis=0, T=1.1)

image = (denormalize(data, 
                     mean=[0.485, 0.456, 0.406], 
                     std=[0.229, 0.224, 0.225]) * 255).transpose(1, 2, 0).astype(np.uint8)
target_img = (target * 225).astype(np.uint8)
pred_img = (pred * 225).astype(np.uint8)

image = Image.fromarray(image)
target_img = Image.fromarray(target_img)
pred_img = Image.fromarray(pred_img)
prob_img = Image.fromarray(np.uint8(cm.coolwarm(prob[cat]) * 255)).convert('RGB')

idx += 1 

# %%
pred_img

# %%
target_img

# %%
image

# %%
