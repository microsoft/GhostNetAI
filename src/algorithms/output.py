# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import json
import os
from pathlib import Path
import tempfile
import zipfile

import numpy as np


class JsonArrayWriter:
    """Publish a JSON array atomically without retaining the prediction corpus."""

    def __init__(self, path):
        self.path = Path(path)
        self.directory = tempfile.TemporaryDirectory(prefix=".ghostnet-", dir=self.path.parent)
        self.temporary = Path(self.directory.name) / "predictions.json"
        self.temporary.write_text("[", encoding="utf-8")
        self.count = 0

    def append(self, record):
        with self.temporary.open("a", encoding="utf-8") as output:
            if self.count:
                output.write(",\n")
            json.dump(record, output, allow_nan=False)
        self.count += 1

    def finalize(self):
        try:
            with self.temporary.open("a", encoding="utf-8") as output:
                output.write("]\n")
            os.replace(self.temporary, self.path)
        finally:
            self.abort()

    def abort(self):
        self.directory.cleanup()


class NpzWriter:
    """Spool batches to disk, then assemble the original NPZ layout a batch at a time."""

    def __init__(self, path):
        self.path = Path(path)
        self.directory = tempfile.TemporaryDirectory(prefix=".ghostnet-", dir=self.path.parent)
        self.count = 0
        self.rows = 0
        self.schema = {}

    def append(self, **arrays):
        arrays = {key: np.asarray(value) for key, value in arrays.items()}
        if not arrays or any(array.ndim == 0 or array.dtype.hasobject for array in arrays.values()):
            raise ValueError("test outputs must be non-object arrays with a batch dimension")
        rows = next(iter(arrays.values())).shape[0]
        if any(array.shape[0] != rows for array in arrays.values()):
            raise ValueError("test output batch dimensions must match")
        if self.count and arrays.keys() != self.schema.keys():
            raise ValueError("test output fields changed between batches")
        for key, array in arrays.items():
            if self.count:
                shape, dtype = self.schema[key]
                if array.shape[1:] != shape or array.dtype.kind != dtype.kind:
                    raise ValueError("test output shape or dtype changed between batches")
                self.schema[key] = (shape, np.result_type(dtype, array.dtype))
            else:
                self.schema[key] = (array.shape[1:], array.dtype)
        np.savez(Path(self.directory.name) / f"batch-{self.count}.npz", **arrays)
        self.count += 1
        self.rows += rows

    def finalize(self):
        temporary = Path(self.directory.name) / "outputs.npz"
        try:
            if not self.count:
                raise ValueError("cannot export an empty test dataset")
            with zipfile.ZipFile(temporary, "w", allowZip64=True) as archive:
                for key, (shape, dtype) in self.schema.items():
                    with archive.open(f"{key}.npy", "w", force_zip64=True) as output:
                        np.lib.format.write_array_header_2_0(output, {
                            "descr": np.lib.format.dtype_to_descr(dtype),
                            "fortran_order": False,
                            "shape": (self.rows,) + shape,
                        })
                        for index in range(self.count):
                            with np.load(Path(self.directory.name) / f"batch-{index}.npz",
                                         allow_pickle=False) as batch:
                                array = np.ascontiguousarray(batch[key], dtype=dtype)
                                output.write(memoryview(array).cast("B"))
            os.replace(temporary, self.path)
        finally:
            self.abort()

    def abort(self):
        self.directory.cleanup()
