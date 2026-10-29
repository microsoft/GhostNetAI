# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import json
import math
import os
from pathlib import Path, PureWindowsPath
import stat
import warnings

from PIL import Image
from pycocotools.coco import COCO

from src.safety import (
    MAX_ANNOTATION_BYTES, MAX_ANNOTATIONS_PER_IMAGE, MAX_DATASET_IMAGES,
    MAX_IMAGE_BYTES, MAX_MASK_WORK_PIXELS, MAX_SEGMENTATION_VALUES,
    bounded_integer, image_dimensions,
)


def dataset_path(root, filename, prefix=""):
    if not isinstance(filename, str) or not filename:
        raise ValueError("image filename must be a nonempty relative path")
    filename = filename.replace("\\", "/")
    if (filename.startswith("/") or PureWindowsPath(filename).drive
            or ":" in filename or ".." in filename.split("/")):
        raise ValueError(f"image filename must stay inside dataset_root: {filename!r}")
    if prefix and filename.startswith(prefix):
        filename = filename[len(prefix):]
    root = Path(root).expanduser().resolve(strict=True)
    path = (root / filename).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"image filename must identify a file inside dataset_root: {filename!r}")
    return path


def check_file(source, maximum, description):
    info = os.fstat(source.fileno())
    if not stat.S_ISREG(info.st_mode) or info.st_size > maximum:
        raise ValueError(f"{description} must be a regular file no larger than {maximum} bytes")


def load_image(path, expected_size=None):
    try:
        with open(path, "rb") as source:
            check_file(source, MAX_IMAGE_BYTES, "image")
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(source) as image:
                    image_dimensions(*image.size)
                    if image.format not in {"JPEG", "PNG", "TIFF", "BMP"}:
                        raise ValueError(f"unsupported image format: {image.format}")
                    if expected_size is not None and image.size != expected_size:
                        raise ValueError("image dimensions do not match annotation dimensions")
                    return image.convert("RGB")
    except (OSError, Image.DecompressionBombWarning, Image.DecompressionBombError) as error:
        raise ValueError(f"cannot safely decode image {path}: {error}") from error


def load_coco(path):
    with Path(path).expanduser().open("rb") as source:
        check_file(source, MAX_ANNOTATION_BYTES, "annotation JSON")
        content = source.read(MAX_ANNOTATION_BYTES + 1)
    if len(content) > MAX_ANNOTATION_BYTES:
        raise ValueError("annotation JSON exceeds its byte limit")
    try:
        data = json.loads(content)
    except (ValueError, RecursionError) as error:
        raise ValueError(f"invalid annotation JSON: {path}") from error
    if not isinstance(data, dict):
        raise ValueError("COCO annotations must be an object")
    images, annotations = data.get("images"), data.get("annotations")
    if not isinstance(images, list) or not isinstance(annotations, list):
        raise ValueError("COCO images and annotations must be lists")
    if len(images) > MAX_DATASET_IMAGES:
        raise ValueError("annotation image count exceeds its limit")
    image_ids = set()
    for image in images:
        if not isinstance(image, dict):
            raise ValueError("COCO image records must be objects")
        image_id = bounded_integer(image.get("id"), "image id", 0, 2**53 - 1)
        if image_id in image_ids:
            raise ValueError("duplicate COCO image id")
        image_ids.add(image_id)
        image_dimensions(image.get("width"), image.get("height"))
        if not isinstance(image.get("file_name"), str):
            raise ValueError("COCO image file_name must be a string")
    counts = {}
    annotation_ids = set()
    for annotation in annotations:
        if not isinstance(annotation, dict):
            raise ValueError("COCO annotation records must be objects")
        annotation_id = bounded_integer(annotation.get("id"), "annotation id", 0, 2**53 - 1)
        image_id = bounded_integer(annotation.get("image_id"), "image id", 0, 2**53 - 1)
        bounded_integer(annotation.get("category_id"), "category id", 0, 2**53 - 1)
        if image_id not in image_ids or annotation_id in annotation_ids:
            raise ValueError("unknown image id or duplicate annotation id")
        annotation_ids.add(annotation_id)
        counts[image_id] = counts.get(image_id, 0) + 1
        if counts[image_id] > MAX_ANNOTATIONS_PER_IMAGE:
            raise ValueError("annotation count per image exceeds its limit")
    coco = COCO()
    coco.dataset = data
    coco.createIndex()
    return coco


def _validate_rle(counts, pixels):
    if isinstance(counts, list):
        runs = counts
    elif isinstance(counts, str):
        runs = []
        index = 0
        while index < len(counts):
            value = shift = 0
            while True:
                if index == len(counts) or shift > 30:
                    raise ValueError("invalid compressed RLE")
                code = ord(counts[index]) - 48
                index += 1
                if not 0 <= code <= 63:
                    raise ValueError("invalid compressed RLE character")
                value |= (code & 31) << shift
                shift += 5
                if not code & 32:
                    if code & 16:
                        value |= -1 << shift
                    break
            if len(runs) > 2:
                value += runs[-2]
            runs.append(value)
    else:
        raise ValueError("RLE counts must be a list or string")
    total = 0
    for run in runs:
        total += bounded_integer(run, "RLE run", 0, pixels)
        if total > pixels:
            raise ValueError("RLE runs exceed image dimensions")
    if total != pixels:
        raise ValueError("RLE runs do not cover image dimensions")


def validate_annotations(annotations, width, height):
    image_dimensions(width, height)
    if len(annotations) > MAX_ANNOTATIONS_PER_IMAGE:
        raise ValueError("annotation count per image exceeds its limit")
    if width * height * len(annotations) > MAX_MASK_WORK_PIXELS:
        raise ValueError("annotation rasterization exceeds its pixel-work limit")
    complexity = 0
    for annotation in annotations:
        segmentation = annotation.get("segmentation")
        if isinstance(segmentation, list):
            if not segmentation:
                raise ValueError("segmentation must contain at least one polygon")
            for polygon in segmentation:
                if not isinstance(polygon, list) or len(polygon) < 6 or len(polygon) % 2:
                    raise ValueError("segmentation polygons need at least three coordinate pairs")
                complexity += len(polygon)
                if complexity > MAX_SEGMENTATION_VALUES:
                    raise ValueError("segmentation complexity exceeds its limit")
                for x, y in zip(polygon[::2], polygon[1::2]):
                    if (isinstance(x, bool) or isinstance(y, bool)
                            or not isinstance(x, (int, float)) or not isinstance(y, (int, float))
                            or not math.isfinite(x) or not math.isfinite(y)
                            or not 0 <= x <= width or not 0 <= y <= height):
                        raise ValueError("segmentation coordinates must be finite and inside the image")
        elif isinstance(segmentation, dict):
            counts = segmentation.get("counts")
            if segmentation.get("size") != [height, width] or not isinstance(counts, (str, list)):
                raise ValueError("RLE size must match image dimensions")
            complexity += len(counts)
            if complexity > MAX_SEGMENTATION_VALUES:
                raise ValueError("segmentation complexity exceeds its limit")
            _validate_rle(counts, width * height)
        else:
            raise ValueError("unsupported segmentation encoding")
