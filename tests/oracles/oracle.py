"""Load unchanged upstream files directly; never import our reproduction."""

import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VENDOR = ROOT / "vendor" / "blockgtq"
COMMIT = "3fe14e7d4b8a6c85402818f7e22052527ed42e93"


def load(name):
    path = VENDOR / f"{name}.py"
    manifest = json.loads((VENDOR / "provenance.json").read_text())
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != manifest["sha256"][path.name]:
        raise ValueError(f"Upstream oracle modified: {path.name}")
    spec = importlib.util.spec_from_file_location(f"upstream_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

