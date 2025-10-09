# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# %%
import os
import numpy as np
from pycocotools.coco import COCO

# %%
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from matplotlib import cm
import cv2

import shapely
from shapely.geometry import Polygon
import pygeoops

# %%
# data_root = os.path.join('~/wwf_ghostnet',
#                          'Ghostnet-labeling-2025-05-21')
# ann_path = os.path.join(data_root, 'ann/2025_05_21_ann.json')

# data_root = os.path.join('~/wwf_ghostnet',
#                          'Ghostnet-labeldata-2024-04-04')
# ann_path = os.path.join(data_root, '2024_04_04_ann.json')

data_root = os.path.join('~/datasets/wwf_ghostnet',
                         '28.01.2025')
ann_path = os.path.join(data_root, '2025_01_28_ann.json')

# %%
ann_coco = COCO(ann_path)

# %%
img_id = 1

# %%
# img_id = 2
img_info = ann_coco.imgs[img_id]
img_anns = ann_coco.loadAnns(ann_coco.getAnnIds(imgIds=img_id))
# img_path = os.path.join(data_root, img_info['file_name'].split('/', 2)[-1])
# img_path = os.path.join(data_root, img_info['file_name'].split('/')[-1])
img_path = os.path.join(data_root, img_info['file_name'].split('28.01.2025/')[-1])

# %%
img_info

# %%
# img = Image.open(img_path).convert("L")
img = Image.open(img_path).convert("RGB")

# %%
img

# %%
draw = ImageDraw.Draw(img)
h, w = img.size

for a in img_anns:
    if 'segmentation' in a:
        color = 'red' if a['category_id'] == 1 else 'blue'
        for seg in a['segmentation']:
            poly = [(int(seg[i] * h) , int(seg[i+1] * w)) for i in range(0, len(seg), 2)]
            draw.polygon(poly, outline=color, width=5)

# %%
img_id += 1
img

# %%
img_anns

# %%
# img.save('./test.png')

# %%
