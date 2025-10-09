# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# %% This File Cuts Puget Sound images into small 1000 X 250 patches for standard scale.
import json
import os
import numpy as np
import cv2
from tqdm import tqdm
from PIL import Image
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils

# %%
np.random.seed(0)

# %%
data_root = os.path.join('~/datasets/wwf_ghostnet',
                         'Ghostnet-labeldata-2024-04-04')
ann_path = os.path.join(data_root, '05042025.json')

cut_data_root = '~/datasets/wwf_seg/PugetSound/2024_04_04_cut'
cut_ann_path = os.path.join(cut_data_root, 'SegAnn/2024_04_04_cut.json')

# %%
def mask_to_ann(mask):
    segs = []
    boxes = []
    areas = []
    contours, _ = cv2.findContours((mask * 225).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        seg = [np.array(c).reshape(-1).tolist()]
        if len(seg[0]) >= 6:
            rles = maskUtils.frPyObjects(seg, 250, 1000)
            rle = maskUtils.merge(rles)
            box = maskUtils.toBbox(rle)
            area = maskUtils.area(rle)
            if area > 220:
                segs.append(seg)
                boxes.append(box)
                areas.append(area.item())
    return segs, boxes, areas

# %%
ann_coco = COCO(ann_path)

cut_ann = {'images': [],
           'annotations': [],
           'categories': [c for _, c in ann_coco.cats.items()]}

cut_img_id = 1
cut_ann_id = 1

for img_id in tqdm(ann_coco.getImgIds()):

    img_info = ann_coco.imgs[img_id]
    img_anns = ann_coco.loadAnns(ann_coco.getAnnIds(imgIds=img_id))
    img_path = os.path.join(data_root, img_info['file_name'].split('/')[-1])
    
    img = Image.open(img_path).convert("L")
    h, w = img.size
    
    mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)
    for i in range(len(img_anns)):
        seg = img_anns[i]['segmentation'][0]
        seg = [int(coord * h) if i % 2 == 0 else int(coord * w) for i, coord in enumerate(seg)]
        img_anns[i]['segmentation'] = [seg]
        mask += ann_coco.annToMask(img_anns[i])
    
    mask = Image.fromarray(np.clip(mask, 0, 1, out=mask))
    
    img = np.array(img.resize((2000, 1000)))
    mask = np.array(mask.resize((2000, 1000)))
    
    img_L = img[:, :1000]
    mask_L = mask[:, :1000] 
    img_R = img[:, 1000:]
    mask_R = mask[:, 1000:] 
    
    for i in range(4):
        i_l = img_L[i*250:(i+1)*250, :]
        m_l = mask_L[i*250:(i+1)*250, :]
        i_r = img_R[i*250:(i+1)*250, :]
        m_r = mask_R[i*250:(i+1)*250, :]
    
        if np.any(m_l > 0):
    
            folder = 'train' if np.random.choice([1, 0], p=[0.8, 0.2]).item() else 'val'
            img_name = os.path.join(folder, img_info['file_name'].split('/')[-1].split('.')[0] + f'_L_{i}.png')
            segs, boxes, areas= mask_to_ann(m_l)
    
            if len(segs) > 0:
            
                cut_ann['images'].append({
                    'id': cut_img_id,
                    'file_name': img_name,
                    'height': 250,
                    'width': 1000 
                })
                for seg, box, area in zip(segs, boxes, areas):
                    cut_ann['annotations'].append({
                        'id': cut_ann_id,
                        'image_id': cut_img_id,
                        'category_id': 1,
                        'segmentation': seg,
                        'area': area,
                        'bbox': box.tolist(),
                    })
                    cut_ann_id += 1
                cut_img_id += 1
    
            Image.fromarray(i_l).save(os.path.join(cut_data_root, img_name))
    
        if np.any(m_r > 0):
            folder = 'train' if np.random.choice([1, 0], p=[0.8, 0.2]).item() else 'val'
            img_name = os.path.join(folder, img_info['file_name'].split('/')[-1].split('.')[0] + f'_R_{i}.png')
            segs, boxes, areas= mask_to_ann(m_r)
    
            if len(segs) > 0:
                cut_ann['images'].append({
                    'id': cut_img_id,
                    'file_name': img_name,
                    'height': 250,
                    'width': 1000 
                })
                for seg, box, area in zip(segs, boxes, areas):
                    cut_ann['annotations'].append({
                        'id': cut_ann_id,
                        'image_id': cut_img_id,
                        'category_id': 1,
                        'segmentation': seg,
                        'area': area,
                        'bbox': box.tolist(),
                    })
                    cut_ann_id += 1
                cut_img_id += 1
    
            folder = 'train' if np.random.choice([1, 0], p=[0.8, 0.2]).item() else 'val'
            Image.fromarray(i_r).save(os.path.join(cut_data_root, img_name))

# %%
with open(cut_ann_path, 'w') as f:
    json.dump(cut_ann, f)

# %%
