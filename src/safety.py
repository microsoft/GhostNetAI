# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import math
import numbers


MAX_IMAGE_PIXELS = 16_000_000
MAX_IMAGE_DIMENSION = 32_768
MAX_IMAGE_BYTES = 64 * 1024 * 1024
MAX_DATASET_IMAGES = 100_000
MAX_ANNOTATION_BYTES = 64 * 1024 * 1024
MAX_ANNOTATIONS_PER_IMAGE = 256
MAX_SEGMENTATION_VALUES = 65_536
MAX_MASK_WORK_PIXELS = 128_000_000
MAX_CLASSES = 256


def bounded_integer(value, name, minimum, maximum):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be in [{minimum}, {maximum}]")
    return int(value)


def bounded_number(value, name, minimum, maximum):
    if (isinstance(value, bool) or not isinstance(value, numbers.Real)
            or not math.isfinite(value) or not minimum <= value <= maximum):
        raise ValueError(f"{name} must be finite and in [{minimum}, {maximum}]")
    return value


def image_dimensions(width, height):
    width = bounded_integer(width, "image width", 1, MAX_IMAGE_DIMENSION)
    height = bounded_integer(height, "image height", 1, MAX_IMAGE_DIMENSION)
    if width * height > MAX_IMAGE_PIXELS:
        raise ValueError(f"image exceeds the {MAX_IMAGE_PIXELS}-pixel limit")
    return width, height
