<!-- # Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. -->
# GhostNet Segmentation

## Example commands: 

### Train
- Every existing hyperparameters are in the configuration files. After a model is trained, a ckpt file will be saved under the `weights` folder.
```
python main.py --config ./configs/ghostseg_de_plain_111824.yaml
```

### Validation 
- Validation will not output npz file for post process. Just outputs evaluation metrics.

```
python main.py --config ./configs/ghostseg_de_plain_111824.yaml --evaluate ./weights/GhostSeg_DE/Plain/GhostSeg_DE_Plain_111824-0-epoch\=2514-valid_mean_IoU\=0.77.ckpt --val
```

### Test
- Test will save an npz file that has images, labels, logits, and all relevent data for post-processing.
```
python main.py --config ./configs/ghostseg_de_plain_111824.yaml --evaluate ./weights/GhostSeg_DE/Plain/GhostSeg_DE_Plain_111824-0-epoch\=2514-valid_mean_IoU\=0.77.ckpt
```

### Prediction
- Prediction only loads a data root and outputs polygon predictions, center lines of the polygons, and centroids of these polygons. All coordinates are in percentages of image widths and heights. 

```
 python main.py --config ./configs/ghostseg_de_plain_111824.yaml --evaluate ./weights/GhostSeg_DE/Plain/GhostSeg_DE_Plain_111824-0-epoch\=2514-valid_mean_IoU\=0.77.ckpt --predict --predict_root ~/datasets/wwf_seg/Germany/val/
```
