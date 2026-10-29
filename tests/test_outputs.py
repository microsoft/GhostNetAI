# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.algorithms.output import JsonArrayWriter, NpzWriter


class OutputTests(unittest.TestCase):
    def test_streamed_json_is_an_array_and_atomic(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.json"
            path.write_text("previous result")
            writer = JsonArrayWriter(path)
            for index in range(1000):
                writer.append({"survey_pic_id": f"image-{index}", "polygon": []})
            self.assertEqual(path.read_text(), "previous result")
            self.assertEqual(writer.count, 1000)
            self.assertFalse(any(isinstance(value, list) for value in vars(writer).values()))
            writer.finalize()
            data = json.loads(path.read_text())
            self.assertEqual(len(data), 1000)
            self.assertEqual(data[-1]["survey_pic_id"], "image-999")
            self.assertFalse(list(Path(directory).glob(".ghostnet-*")))
            writer.abort()

    def test_empty_json_and_failed_output_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.json"
            writer = JsonArrayWriter(path)
            writer.finalize()
            self.assertEqual(json.loads(path.read_text()), [])
            writer = JsonArrayWriter(path)
            try:
                with self.assertRaises(ValueError):
                    writer.append({"confidence": float("nan")})
            finally:
                writer.abort()
            self.assertEqual(json.loads(path.read_text()), [])
            self.assertFalse(list(Path(directory).glob(".ghostnet-*")))

    def test_npz_preserves_keys_shapes_values_and_variable_length_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "outputs.npz"
            writer = NpzWriter(path)
            expected = []
            for index, size in enumerate((2, 2, 1)):
                arrays = {
                    "data": np.full((size, 3, 4, 4), index, dtype=np.float32),
                    "preds": np.ones((size, 4, 4), dtype=np.int64),
                    "targets": np.zeros((size, 4, 4), dtype=np.uint8),
                    "logits": np.full((size, 2, 4, 4), index, dtype=np.float32),
                    "file_ids": np.asarray(["a" * (index + 1) + ".jpg"] * size),
                }
                writer.append(**arrays)
                expected.append(arrays)
            self.assertFalse(path.exists())
            self.assertEqual(writer.rows, 5)
            self.assertFalse(any(isinstance(value, (list, np.ndarray)) for value in vars(writer).values()))
            writer.finalize()
            with np.load(path, allow_pickle=False) as actual:
                self.assertEqual(set(actual.files), set(expected[0]))
                for key in actual.files:
                    np.testing.assert_array_equal(actual[key], np.concatenate([batch[key] for batch in expected]))
            self.assertFalse(list(Path(directory).glob(".ghostnet-*")))

    def test_npz_rejects_inconsistent_batches_and_cleans_up(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "outputs.npz"
            writer = NpzWriter(path)
            try:
                writer.append(data=np.zeros((1, 2)))
                with self.assertRaises(ValueError):
                    writer.append(data=np.zeros((1, 3)))
                with self.assertRaises(ValueError):
                    writer.append(data=np.array([object()]))
                with self.assertRaises(ValueError):
                    writer.append(other=np.zeros((1, 2)))
            finally:
                writer.abort()
            self.assertFalse(path.exists())
            self.assertFalse(list(Path(directory).glob(".ghostnet-*")))


if __name__ == "__main__":
    unittest.main()
