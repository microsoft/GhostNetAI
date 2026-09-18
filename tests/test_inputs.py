# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import yaml
from PIL import Image
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO

from src.config import load_config, select_component
from src.datasets.datasets import GhostSegDS, GhostSegDS_Pred
from src.datasets.ext_transforms import ExtRandomCrop, ExtRandomScale, ExtScale
from src.datasets.input_validation import dataset_path, load_coco, load_image, validate_annotations
from src.safety import MAX_IMAGE_PIXELS, image_dimensions


class ConfigTests(unittest.TestCase):
    def test_bundled_configs(self):
        for path in (Path(__file__).resolve().parents[1] / "configs").glob("*.yaml"):
            with self.subTest(path=path):
                self.assertEqual(load_config(path).num_classes, 2)

    def test_unsafe_yaml_and_non_mapping_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            for content in ("!!python/name:builtins.eval", "[]", "null", "x" * (1024 * 1024 + 1)):
                path.write_text(content)
                with self.subTest(content=content[:30]), self.assertRaises((ValueError, yaml.YAMLError)) as result:
                    load_config(path)
                self.assertIsNotNone(result.exception)

    def test_explicit_component_selection(self):
        constructor = Mock()
        self.assertIs(select_component({"Plain": constructor}, "Plain", "algorithm"), constructor)
        for name in ("__builtins__", "utils", "torch", None, []):
            with self.subTest(name=name), self.assertRaises(ValueError):
                select_component({"Plain": constructor}, name, "algorithm")
        constructor.assert_not_called()


class ImageInputTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "val").mkdir()
        self.image = self.root / "val" / "image.jpg"
        Image.new("RGB", (8, 6)).save(self.image)

    def test_root_confinement_and_legacy_prefix(self):
        self.assertEqual(dataset_path(self.root, "wwf_ann/Germany/val/image.jpg",
                                      "wwf_ann/Germany/"), self.image.resolve())
        self.assertEqual(dataset_path(self.root, r"val\image.jpg"), self.image.resolve())
        for filename in ("../outside.jpg", "val/../../outside.jpg", "/etc/passwd",
                         r"C:\outside.jpg", r"C:outside.jpg", r"\\server\share\file",
                         r"val\..\outside.jpg", "val/image.jpg:stream"):
            with self.subTest(filename=filename), self.assertRaises(ValueError):
                dataset_path(self.root, filename)

    def test_symlink_escape(self):
        with tempfile.TemporaryDirectory() as outside:
            target = Path(outside) / "outside.jpg"
            Image.new("RGB", (4, 4)).save(target)
            link = self.root / "val" / "link.jpg"
            try:
                link.symlink_to(target)
            except OSError as error:
                self.skipTest(f"symlinks unavailable: {error}")
            with self.assertRaises(ValueError):
                dataset_path(self.root, "val/link.jpg")

    def test_resolved_path_must_remain_under_root(self):
        root = self.root.resolve()
        with patch.object(Path, "resolve", side_effect=[root, root.parent / "outside.jpg"]):
            with self.assertRaisesRegex(ValueError, "inside dataset_root"):
                dataset_path(root, "val/link.jpg")

    def test_image_and_annotation_size_match(self):
        self.assertEqual(load_image(self.image, (8, 6)).size, (8, 6))
        with self.assertRaisesRegex(ValueError, "dimensions"):
            load_image(self.image, (7, 6))
        invalid = self.root / "not-an-image.jpg"
        invalid.write_text("not an image")
        with self.assertRaisesRegex(ValueError, "decode"):
            load_image(invalid)

    def test_oversized_header_rejected_before_decode(self):
        image = Mock(size=(8000, 8000), format="JPEG")
        with patch("src.datasets.input_validation.Image.open") as opened:
            opened.return_value.__enter__.return_value = image
            with self.assertRaisesRegex(ValueError, "pixel"):
                load_image(self.image)
        image.convert.assert_not_called()

    def test_encoded_image_byte_limit(self):
        with patch("src.datasets.input_validation.MAX_IMAGE_BYTES", self.image.stat().st_size - 1):
            with patch("src.datasets.input_validation.Image.open") as opened:
                with self.assertRaisesRegex(ValueError, "regular file"):
                    load_image(self.image)
                opened.assert_not_called()

    def test_dimension_boundaries(self):
        self.assertEqual(image_dimensions(4000, 4000), (4000, 4000))
        self.assertEqual(4000 * 4000, MAX_IMAGE_PIXELS)
        for width, height in ((4001, 4000), (0, 1), (-1, 1), (1.5, 1), (True, 1), (32769, 1)):
            with self.subTest(size=(width, height)), self.assertRaises(ValueError):
                image_dimensions(width, height)

    def test_prediction_loader_is_bounded_and_decodes(self):
        dataset = GhostSegDS_Pred(self.root)
        image, path, size = dataset[0]
        self.assertEqual(image.size, (8, 6))
        self.assertEqual(Path(path), self.image.resolve())
        self.assertEqual(size.tolist(), [8, 6])
        with patch("src.datasets.datasets.MAX_DATASET_IMAGES", 0):
            with self.assertRaisesRegex(ValueError, "image count"):
                GhostSegDS_Pred(self.root)


class AnnotationTests(unittest.TestCase):
    def annotation(self, segmentation=None):
        return {"id": 1, "image_id": 1, "category_id": 1,
                "segmentation": segmentation if segmentation is not None else [[0, 0, 4, 0, 4, 4, 0, 4]]}

    def test_annotation_budgets(self):
        annotation = self.annotation()
        with self.assertRaisesRegex(ValueError, "count"):
            validate_annotations([annotation] * 257, 8, 6)
        with self.assertRaisesRegex(ValueError, "pixel-work"):
            validate_annotations([annotation] * 9, 4000, 4000)
        with self.assertRaisesRegex(ValueError, "complexity"):
            validate_annotations([self.annotation([[0, 0] * 32769])], 8, 6)
        with self.assertRaisesRegex(ValueError, "at least one"):
            validate_annotations([self.annotation([])], 8, 6)
        for polygon in ([0, 0, 1], [0, 0, float("nan"), 1, 2, 2],
                        [0, 0, 9, 1, 2, 2]):
            with self.subTest(polygon=polygon), self.assertRaises(ValueError):
                validate_annotations([self.annotation([polygon])], 8, 6)

    def test_rle_is_validated_before_native_decode(self):
        mask = np.zeros((6, 8), dtype=np.uint8, order="F")
        mask[1:4, 2:6] = 1
        encoded = mask_utils.encode(mask)
        encoded["counts"] = encoded["counts"].decode("ascii")
        validate_annotations([self.annotation(encoded)], 8, 6)
        validate_annotations([self.annotation({"size": [6, 8], "counts": [20, 10, 18]})], 8, 6)
        for segmentation in ({"size": [6, 7], "counts": [48]},
                             {"size": [6, 8], "counts": [49]},
                             {"size": [6, 8], "counts": [47]},
                             {"size": [6, 8], "counts": [-1, 49]},
                             {"size": [6, 8], "counts": "zzzzzzzz"},
                             {"size": [6, 8], "counts": "P"}):
            with self.subTest(segmentation=segmentation), self.assertRaises(ValueError):
                validate_annotations([self.annotation(segmentation)], 8, 6)

    def test_coco_limits_before_indexing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "annotations.json"
            data = {"images": [{"id": 1, "file_name": "val/image.jpg", "width": 8, "height": 6}],
                    "annotations": [self.annotation()]}
            path.write_text(json.dumps(data))
            self.assertEqual(len(load_coco(path).imgs), 1)
            with patch("src.datasets.input_validation.MAX_ANNOTATION_BYTES", 8):
                with patch.object(COCO, "createIndex") as create_index:
                    with self.assertRaisesRegex(ValueError, "regular file"):
                        load_coco(path)
                    create_index.assert_not_called()
            data["images"][0]["width"] = 100000
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "width"):
                load_coco(path)
            data["images"][0]["width"] = 8
            data["annotations"] = [dict(self.annotation(), id=i) for i in range(257)]
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "count"):
                load_coco(path)

    def test_dataset_masks_do_not_overflow_and_reject_before_rasterization(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (8, 6)).save(root / "val.jpg")
            coco = COCO()
            coco.dataset = {
                "images": [{"id": 1, "file_name": "val.jpg", "width": 8, "height": 6}],
                "annotations": [dict(self.annotation(), id=i) for i in range(256)],
            }
            coco.createIndex()
            dataset = GhostSegDS(coco, root, dset="val", path_replace="")
            _, mask, _ = dataset[0]
            self.assertEqual(np.asarray(mask)[1, 1], 1)
            coco.imgToAnns[1].append(self.annotation())
            with patch.object(coco, "annToMask") as decode:
                with self.assertRaisesRegex(ValueError, "count"):
                    dataset[0]
                decode.assert_not_called()


class TransformTests(unittest.TestCase):
    def test_mismatched_pairs_always_rejected(self):
        image, label = Image.new("RGB", (8, 6)), Image.new("L", (7, 6))
        for transform in (ExtRandomCrop(4), ExtRandomScale((0.5, 2)), ExtScale(1)):
            with self.subTest(transform=type(transform).__name__), self.assertRaisesRegex(ValueError, "match"):
                transform(image, label)

    def test_scale_validation_and_rounding(self):
        for scale in (0, -1, float("inf"), float("nan"), 100, True):
            with self.subTest(scale=scale), self.assertRaises(ValueError):
                ExtRandomScale((scale, scale))
            with self.subTest(scale=scale), self.assertRaises(ValueError):
                ExtScale(scale)
        with self.assertRaises(ValueError):
            ExtRandomScale((2, 1))
        with self.assertRaises(ValueError):
            ExtRandomScale((0.1, 0.1))(Image.new("RGB", (1, 1)), Image.new("L", (1, 1)))
        image, label = ExtRandomScale((2, 2))(Image.new("RGB", (8, 6)), Image.new("L", (8, 6)))
        self.assertEqual(image.size, (16, 12))
        self.assertEqual(label.size, image.size)

    def test_crop_preconditions_and_padding(self):
        image, label = Image.new("RGB", (3, 2)), Image.new("L", (3, 2))
        with self.assertRaisesRegex(ValueError, "pad_if_needed"):
            ExtRandomCrop((6, 8))(image, label)
        cropped, target = ExtRandomCrop((6, 8), pad_if_needed=True)(image, label)
        self.assertEqual(cropped.size, (8, 6))
        self.assertEqual(target.size, cropped.size)
        self.assertEqual(ExtRandomCrop.get_params(cropped, (6, 8)), (0, 0, 6, 8))
        for size in (0, -1, (1, 0), (1.5, 2), (4001, 4000)):
            with self.subTest(size=size), self.assertRaises(ValueError):
                ExtRandomCrop(size)


if __name__ == "__main__":
    unittest.main()
