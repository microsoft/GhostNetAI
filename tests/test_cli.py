# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import unittest
from unittest.mock import Mock, patch

from munch import Munch

import main


class CliTests(unittest.TestCase):
    def run_main(self, *, evaluate=None, val=False, predict=False, weights_init=None):
        conf = Munch(dataset_name="GhostSeg_DE", algorithm="Plain", conf_id="test",
                     num_iters=1, log_interval=1, batch_size=1, weights_init=weights_init)
        learner, dataset = Mock(), Mock()
        with patch.object(main, "load_config", return_value=conf), \
                patch.dict(main.algorithms.ALGORITHMS, {"Plain": Mock(return_value=learner)}), \
                patch.dict(main.datasets.DATASETS, {"GhostSeg_DE": Mock(return_value=dataset)}), \
                patch.object(main, "load_weights", return_value=Mock(missing_keys=[], unexpected_keys=[])) as load, \
                patch.object(main, "CSVLogger"), \
                patch.object(main.L, "Trainer") as trainer:
            main.main(accelerator="cpu", gpus="not-used-on-cpu", logger_type="csv",
                      evaluate=evaluate, val=val, predict=predict)
        return load, trainer.return_value, learner, dataset

    def test_all_evaluation_modes_use_preloaded_weights(self):
        for mode in ("validate", "test", "predict"):
            with self.subTest(mode=mode):
                load, trainer, learner, dataset = self.run_main(
                    evaluate="weights.ckpt", val=mode == "validate", predict=mode == "predict")
                load.assert_called_once_with(learner, "weights.ckpt")
                kwargs = {"datamodule": dataset}
                if mode == "predict":
                    kwargs["return_predictions"] = False
                getattr(trainer, mode).assert_called_once_with(learner, **kwargs)
                self.assertNotIn("ckpt_path", getattr(trainer, mode).call_args.kwargs)

    def test_training_initial_weights_use_same_safe_loader(self):
        load, trainer, learner, dataset = self.run_main(weights_init="initial.ckpt")
        load.assert_called_once_with(learner, "initial.ckpt", strict=False)
        trainer.fit.assert_called_once_with(learner, datamodule=dataset)

    def test_unknown_dispatch_is_rejected_before_construction(self):
        conf = Munch(dataset_name="__builtins__", algorithm="Plain", batch_size=1)
        with patch.object(main, "load_config", return_value=conf), \
                patch.object(main.L, "Trainer") as trainer:
            with self.assertRaisesRegex(ValueError, "unsupported dataset"):
                main.main(accelerator="cpu")
        trainer.assert_not_called()


if __name__ == "__main__":
    unittest.main()
