# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import contextlib
import pickle
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import cv2
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from munch import Munch
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src import models
from src.algorithms.la import LA
from src.algorithms.plain import Plain, MAX_VALIDATION_BYTES, MAX_VALIDATION_SAMPLES
from src.algorithms.utils import StreamSegMetrics, colorize, prediction_outputs, softmax
from src.checkpoints import load_weights
from src.config import validate_config


class TinyNet(nn.Module):
    def __init__(self, num_cls):
        super().__init__()
        self.feature = nn.Conv2d(3, 4, 1)
        self.classifier = nn.Conv2d(4, num_cls, 1)

    def forward(self, data):
        return self.classifier(self.feature(data))

    def setup_criteria(self):
        self.criterion_seg = nn.CrossEntropyLoss(ignore_index=255)


def config(**overrides):
    values = dict(algorithm="Plain", model_name="DeepLabV3_ResNet50", num_classes=2,
                  weight_decay_feature=0.0004, weight_decay_classifier=0.0004, tau=1,
                  lr_feature=0.001, lr_classifier=0.01, momentum_feature=0.9,
                  momentum_classifier=0.9, num_iters=2, weights_init=None)
    values.update(overrides)
    return Munch(values)


class MetricTests(unittest.TestCase):
    def test_class_count_rejected_before_allocation(self):
        for count in (-1, 0, 1, 257, 10**9, 2.5, True):
            with self.subTest(count=count), patch("src.algorithms.utils.np.zeros") as allocate:
                with self.assertRaises(ValueError):
                    StreamSegMetrics(count)
                allocate.assert_not_called()
        self.assertEqual(StreamSegMetrics(256).confusion_matrix.shape, (256, 256))

    def test_invalid_predictions_rejected_before_bincount(self):
        for prediction in (np.array([[-1]]), np.array([[2]]), np.array([[10**9]]),
                           np.array([[1.5]]), np.array([[float("nan")]])):
            with self.subTest(prediction=prediction), patch("src.algorithms.utils.np.bincount") as count:
                with self.assertRaises(ValueError):
                    StreamSegMetrics(2).update(np.array([[0]]), prediction)
                count.assert_not_called()
        with self.assertRaises(ValueError):
            StreamSegMetrics(2).update(np.zeros((1, 2), dtype=int), np.zeros((2, 2), dtype=int))

    def test_empty_and_absent_classes(self):
        metrics = StreamSegMetrics(3)
        with self.assertRaisesRegex(ValueError, "without valid"):
            metrics.get_results()
        metrics.update(np.array([[0, 0, 255, -1]]), np.array([[0, 1, 2, 0]]))
        with np.errstate(divide="raise", invalid="raise"):
            scores = metrics.get_results()
        self.assertEqual(scores["Overall Acc"], 0.5)
        self.assertEqual(scores["Mean Acc"], 0.5)
        self.assertTrue(np.isfinite(scores["Per Cls Acc"]).all())
        np.testing.assert_array_equal(metrics.confusion_matrix, [[1, 1, 0], [0, 0, 0], [0, 0, 0]])

    def test_multiclass_color_and_ignore_labels(self):
        colored = colorize(np.array([[0, 1, 2, 255, -1]]), 3)
        np.testing.assert_array_equal(colored[0, :2], [[0, 0, 0], [255, 255, 255]])
        np.testing.assert_array_equal(colored[0, 3:], [[128, 128, 128], [128, 128, 128]])
        with self.assertRaises(ValueError):
            colorize(np.array([[3]]), 3)


class GeometryTests(unittest.TestCase):
    def test_empty_and_simple_prediction(self):
        pred = np.zeros((100, 200), dtype=np.uint8)
        probs = np.full(pred.shape, 0.9)
        self.assertEqual(prediction_outputs(pred, probs, (200, 100)), ([], [], [], [], []))
        cv2.ellipse(pred, (100, 50), (60, 20), 0, 0, 360, 1, -1)
        polygons, centroids, lines, means, medians = prediction_outputs(pred, probs, (200, 100))
        self.assertEqual(len(polygons), 1)
        self.assertEqual(len(centroids), 1)
        self.assertEqual(len(lines), 1)
        self.assertAlmostEqual(means[0], 0.9)
        self.assertAlmostEqual(medians[0], 0.9)
        self.assertTrue(all(0 <= value <= 1 for point in polygons[0] for value in point))

    def test_complexity_rejected_before_centerline(self):
        cases = ([np.zeros((11, 1, 2), dtype=np.int32)] * 129,
                 [np.zeros((513, 1, 2), dtype=np.int32)],
                 [np.zeros((512, 1, 2), dtype=np.int32)] * 9)
        for contours in cases:
            with self.subTest(count=len(contours)), \
                    patch("src.algorithms.utils.cv2.findContours", return_value=(contours, None)), \
                    patch("src.algorithms.utils.pygeoops.centerline") as centerline:
                with self.assertRaisesRegex(ValueError, "limits"):
                    prediction_outputs(np.zeros((16, 16)), np.ones((16, 16)))
                centerline.assert_not_called()

    def test_geometry_input_validation(self):
        for pred, probs in ((np.zeros((2, 2)), np.zeros((2, 3))),
                            (np.full((2, 2), 2), np.ones((2, 2))),
                            (np.zeros((2, 2)), np.full((2, 2), float("nan")))):
            with self.subTest(pred=pred), self.assertRaises(ValueError):
                prediction_outputs(pred, probs)

    def test_stable_softmax(self):
        probabilities = softmax(np.array([[10000., -10000.], [10001., -10001.]]))
        self.assertTrue(np.isfinite(probabilities).all())
        np.testing.assert_allclose(probabilities.sum(axis=0), [1, 1])


class CheckpointTests(unittest.TestCase):
    def test_safe_state_dict_and_lightning_checkpoint(self):
        source, target = nn.Linear(2, 2), nn.Linear(2, 2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.ckpt"
            for checkpoint in (source.state_dict(), {"state_dict": source.state_dict(), "epoch": 3}):
                torch.save(checkpoint, path)
                load_weights(target, path)
                for name, value in source.state_dict().items():
                    torch.testing.assert_close(target.state_dict()[name], value)
            torch.save({"state_dict": {"weight": "not a tensor"}}, path)
            with self.assertRaises(ValueError):
                load_weights(target, path)

    def test_pickle_code_is_not_executed(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "executed.txt"

            class Payload:
                def __reduce__(self):
                    return eval, (f"__import__('pathlib').Path({str(marker)!r}).write_text('executed')",)

            path = Path(directory) / "untrusted.ckpt"
            torch.save({"state_dict": nn.Linear(2, 2).state_dict(), "extra": Payload()}, path)
            with self.assertRaises(pickle.UnpicklingError):
                load_weights(nn.Linear(2, 2), path)
            self.assertFalse(marker.exists())

    def test_loader_explicitly_requests_safe_cpu_loading(self):
        model = nn.Linear(2, 2)
        with patch("src.checkpoints.torch.load", return_value=model.state_dict()) as load:
            load_weights(model, "weights.ckpt")
        self.assertEqual(load.call_args.kwargs, {"map_location": "cpu", "weights_only": True})


class AlgorithmTests(unittest.TestCase):
    def setUp(self):
        registry = patch.dict(models.MODELS, {"DeepLabV3_ResNet50": TinyNet})
        registry.start()
        self.addCleanup(registry.stop)

    def test_weight_decay_validation(self):
        for value in (-0.1, float("nan"), float("inf"), 1e100, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                Plain(config(weight_decay_feature=value))
        for value in (0, 0.0004, 1):
            validate_config(config(weight_decay_feature=value))
        with self.assertRaisesRegex(ValueError, "two classes"):
            LA(config(algorithm="LA", num_classes=3))

    def test_logit_adjustment_cpu_and_checkpoint_compatibility(self):
        learner = LA(config(algorithm="LA"))
        learner.log = Mock()
        learner.on_train_start()
        data, labels = torch.randn(2, 3, 4, 4), torch.zeros((2, 4, 4), dtype=torch.long)
        loss = learner.training_step((data, labels, ["a", "b"]), 0)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(loss.device.type, "cpu")
        loss.backward()
        self.assertNotIn("adjustments", learner.state_dict())
        self.assertIn("adjustments", dict(learner.named_buffers()))

    def test_validation_retention_is_bounded_and_metrics_cover_every_sample(self):
        learner = Plain(config())
        learner.on_validation_start()
        batch = (torch.randn(4, 3, 4, 4), torch.zeros((4, 4, 4), dtype=torch.long), ["x"] * 4)
        for index in range(20):
            learner.validation_step(batch, index)
        self.assertEqual(len(learner.val_st_outs), MAX_VALIDATION_SAMPLES)
        self.assertLessEqual(learner.validation_bytes, MAX_VALIDATION_BYTES)
        self.assertEqual(learner.metrics.confusion_matrix.sum(), 20 * 4 * 4 * 4)
        self.assertTrue(all(array.flags.owndata for sample in learner.val_st_outs for array in sample))

    def test_validation_byte_limit_prevents_copies(self):
        learner = Plain(config())
        learner.on_validation_start()
        with patch("src.algorithms.plain.MAX_VALIDATION_BYTES", 1):
            learner.validation_step((torch.zeros(1, 3, 4, 4), torch.zeros(1, 4, 4, dtype=torch.long), ["x"]), 0)
        self.assertEqual(learner.val_st_outs, [])
        self.assertEqual(learner.validation_bytes, 0)
        self.assertEqual(learner.metrics.confusion_matrix.sum(), 16)

    def test_multiclass_tensorboard_previews(self):
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

        with tempfile.TemporaryDirectory() as directory:
            logger = TensorBoardLogger(directory)
            trainer = L.Trainer(accelerator="cpu", devices=1, logger=logger,
                                enable_checkpointing=False, enable_progress_bar=False,
                                enable_model_summary=False)
            learner = Plain(config(num_classes=3))
            samples = [(torch.zeros(3, 4, 4), torch.full((4, 4), 2, dtype=torch.long), "image.jpg")]
            trainer.validate(learner, dataloaders=DataLoader(samples))
            logger.experiment.flush()
            events = EventAccumulator(logger.log_dir).Reload()
            self.assertEqual(set(events.Tags()["images"]),
                             {"validation/0_data", "validation/0_target", "validation/0_pred"})
            logger.experiment.close()
            self.assertEqual(learner.val_st_outs, [])

    def test_actual_lightning_cpu_lifecycle_and_exports(self):
        class Samples(Dataset):
            def __len__(self):
                return 3

            def __getitem__(self, index):
                return torch.zeros(3, 8, 8), torch.zeros(8, 8, dtype=torch.long), f"image-{index}.jpg"

        class Predictions(Samples):
            def __getitem__(self, index):
                return torch.zeros(3, 8, 8), f"image-{index}.jpg", torch.tensor([8, 8])

        with tempfile.TemporaryDirectory() as directory, contextlib.chdir(directory):
            path = Path(directory) / "weights.ckpt"
            learner = LA(config(algorithm="LA", evaluate=str(path)))
            trainer = L.Trainer(accelerator="cpu", devices=1, logger=False,
                                enable_checkpointing=False, enable_progress_bar=False,
                                enable_model_summary=False, max_epochs=1, num_sanity_val_steps=0)
            loader = DataLoader(Samples(), batch_size=2)
            trainer.fit(learner, train_dataloaders=loader, val_dataloaders=loader)
            trainer.save_checkpoint(path)
            restored = LA(config(algorithm="LA", evaluate=str(path)))
            load_weights(restored, path)
            trainer.validate(restored, dataloaders=loader)
            self.assertEqual(restored.val_st_outs, [])
            self.assertLessEqual(len(list((Path(directory) / "log" / "temp_imgs").glob("*.png"))), 30)
            trainer.test(restored, dataloaders=loader)
            with np.load(path.with_suffix(".npz"), allow_pickle=False) as outputs:
                self.assertEqual(outputs["data"].shape, (3, 3, 8, 8))
                self.assertEqual(outputs["logits"].shape, (3, 2, 8, 8))
                self.assertEqual(outputs["file_ids"].tolist(), [f"image-{i}.jpg" for i in range(3)])
            with torch.no_grad():
                restored.net.classifier.weight.zero_()
                restored.net.classifier.bias.copy_(torch.tensor([1., 0.]))
            trainer.predict(restored, dataloaders=DataLoader(Predictions(), batch_size=2),
                            return_predictions=False)
            import json
            records = json.loads(path.with_name("weights_predict.json").read_text())
            self.assertEqual(len(records), 3)
            self.assertEqual(records[0]["original_image_size"], [8, 8])
            self.assertEqual(records[0]["marker_bounding_polygon"], [])
            self.assertFalse(list(Path(directory).glob(".ghostnet-*")))


if __name__ == "__main__":
    unittest.main()
