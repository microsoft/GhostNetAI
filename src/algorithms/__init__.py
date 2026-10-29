# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
from .plain import Plain
from .la import LA

ALGORITHMS = {"Plain": Plain, "LA": LA}
__all__ = ["Plain", "LA"]
