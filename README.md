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
- Test saves an NPZ file with `data`, `preds`, `targets`, `logits`, and `file_ids`
  for post-processing. Batches are spooled to temporary files beside the checkpoint,
  then assembled without keeping the test corpus in memory. Allow disk space for
  both the temporary batches and the completed archive. The destination is replaced
  only after export succeeds.
```
python main.py --config ./configs/ghostseg_de_plain_111824.yaml --evaluate ./weights/GhostSeg_DE/Plain/GhostSeg_DE_Plain_111824-0-epoch\=2514-valid_mean_IoU\=0.77.ckpt
```

### Prediction
- Prediction only loads a data root and outputs polygon predictions, center lines of the polygons, and centroids of these polygons. All coordinates are in percentages of image widths and heights. 

```
 python main.py --config ./configs/ghostseg_de_plain_111824.yaml --evaluate ./weights/GhostSeg_DE/Plain/GhostSeg_DE_Plain_111824-0-epoch\=2514-valid_mean_IoU\=0.77.ckpt --predict --predict-root ~/datasets/wwf_seg/Germany/val/
```

## Input validation and resource limits

Install `requirements.txt` with Python 3.12. The Torch and Torchvision pins
are a matching pair, and `pygeoops` is required for polygon predictions.
For machines without CUDA, pass `--accelerator cpu`.

Configuration uses standard safe YAML, limited to 1 MiB. Dataset, algorithm, and
model names must match the explicitly registered components. `num_classes` must
be an integer from 2 through 256. Weight decay must be finite and between 0 and 1.
Logit adjustment requires two classes and a finite `tau` between 0 and 100;
polygon prediction also requires two classes.

Both `--evaluate` and `weights_init` load tensor state dictionaries with
`weights_only=True` on the CPU. Standard Lightning checkpoints and raw tensor
state dictionaries are supported; checkpoints requiring arbitrary Python objects
are rejected, with no unsafe fallback. Evaluation restores model weights, not
trainer or optimizer state. Use checkpoints from sources you trust: restricted
deserialization does not validate model quality or guarantee that arbitrary tensor
payloads fit available memory.

Dataset filenames must remain within `dataset_root`, including after resolving
symlinks. Absolute paths, parent traversal, and Windows drive/stream paths are
rejected. JPEG, PNG, TIFF, and BMP images are checked before decoding: at most
64 MiB encoded, 16 million pixels, and 32,768 pixels on either axis. COCO dimensions
must match the decoded image. Prediction discovery does not follow directory
symlinks. Datasets are limited to 100,000 images.

COCO JSON files are limited to 64 MiB before parsing. Each image may have at most
256 annotations and 65,536 segmentation coordinate values or RLE units in total.
Rasterization is limited to 128 million pixels of work (image pixels multiplied
by annotation count). Polygon coordinates and RLE sizes/counts are validated
before native mask decoding. Invalid or over-budget inputs raise descriptive
errors rather than being silently skipped.

Scale factors must be finite and in `[0.000001, 8]`; resulting dimensions must
remain positive and within the image limits. Image/label dimensions must match.
Crops larger than an image require `pad_if_needed=True`. These checks also apply
when Python runs with `-O`.

Validation metrics cover the full dataset, but previews retain at most 10 samples
and 128 MiB per process. Labels for all configured classes can be visualized;
ignored labels (`255` outside the class range, and `-1`) are gray. Empty metric
inputs raise an error instead of producing NaN scores; absent classes have
zero per-class accuracy and do not contribute to mean accuracy.

Prediction geometry is limited to 128 contours, 512 vertices per contour, and
4,096 vertices in total per image. Automatic centerline densification is disabled
to keep its input bounded. Invalid polygons and over-budget geometry are rejected.
Prediction records are streamed to a temporary JSON array and published atomically
with the existing output schema. Distributed test/prediction jobs use `_rankN`
filename suffixes so workers do not overwrite one another.

## Regression tests

After installing the requirements, run the CPU-only suite (no model download):

```sh
python -m unittest discover -s tests -v
python -O -m unittest discover -s tests -v
```
