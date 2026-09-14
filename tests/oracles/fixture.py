"""Version 1: JSON array descriptors plus a little-endian binary payload."""

import hashlib
import json
from pathlib import Path

import numpy as np

DTYPES = {"f32": np.dtype("<f4"), "u32": np.dtype("<u4"), "u8": np.dtype("u1"),
          "f16": np.dtype("<f2"), "u16": np.dtype("<u2")}


def write_fixture(path, metadata, arrays):
    path = Path(path)
    payload = bytearray()
    descriptors = {}
    for name, (dtype, data) in arrays.items():
        arr = np.asarray(data, dtype=DTYPES[dtype])
        payload.extend(b"\0" * ((-len(payload)) % 4))
        descriptors[name] = dict(dtype=dtype, shape=list(arr.shape),
                                 offset=len(payload), nbytes=arr.nbytes)
        payload.extend(arr.tobytes(order="C"))
    payload.extend(b"\0" * ((-len(payload)) % 4))
    path.parent.mkdir(parents=True, exist_ok=True)
    binary = path.with_suffix(".bin")
    binary.write_bytes(payload)
    manifest = dict(metadata, version=metadata.get("version", 1), endianness="little", binary=binary.name,
                    byte_length=len(payload), sha256=hashlib.sha256(payload).hexdigest(),
                    arrays=descriptors)
    path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_fixture(path):
    path = Path(path)
    meta = json.loads(path.read_text(encoding="utf-8"))
    if meta["version"] not in (1, 2) or meta["endianness"] != "little":
        raise ValueError("Unsupported fixture format")
    payload = path.with_name(meta["binary"]).read_bytes()
    if len(payload) != meta["byte_length"] or hashlib.sha256(payload).hexdigest() != meta["sha256"]:
        raise ValueError("Fixture payload length/hash mismatch")
    arrays = {}
    for name, desc in meta["arrays"].items():
        dtype = DTYPES[desc["dtype"]]
        expected = int(np.prod(desc["shape"])) * dtype.itemsize
        if desc["offset"] % 4 or desc["nbytes"] != expected or desc["offset"] + expected > len(payload):
            raise ValueError(f"Invalid array descriptor: {name}")
        arrays[name] = np.frombuffer(payload, dtype, expected // dtype.itemsize,
                                   desc["offset"]).reshape(desc["shape"]).copy()
    return meta, arrays
