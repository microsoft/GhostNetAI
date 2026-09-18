# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
from pathlib import Path

import yaml
from munch import Munch

from .safety import MAX_CLASSES, bounded_integer, bounded_number


def load_config(path):
    with Path(path).expanduser().open("rb") as source:
        content = source.read(1024 * 1024 + 1)
    if len(content) > 1024 * 1024:
        raise ValueError("configuration exceeds the 1 MiB limit")
    config = yaml.safe_load(content)
    if not isinstance(config, dict) or not all(isinstance(key, str) for key in config):
        raise ValueError("configuration must be a mapping with string keys")
    validate_config(config)
    return Munch(config)


def validate_config(config):
    bounded_integer(config.get("num_classes"), "num_classes", 2, MAX_CLASSES)
    for key in ("weight_decay_feature", "weight_decay_classifier"):
        bounded_number(config.get(key), key, 0, 1)
    if config.get("algorithm") == "LA":
        if config["num_classes"] != 2:
            raise ValueError("LA requires exactly two classes")
        bounded_number(config.get("tau"), "tau", 0, 100)


def select_component(registry, name, kind):
    if not isinstance(name, str) or name not in registry:
        raise ValueError(f"unsupported {kind} {name!r}; expected one of {', '.join(registry)}")
    return registry[name]
