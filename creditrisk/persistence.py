"""Saving and loading a trained model safely.

Saved files (in the model directory):
    model.joblib     the fitted pipeline
    reference.csv    cleaned training records
    metadata.json    metrics, settings, valid ranges, library versions and a SHA-256 of the files above

Loading verifies the SHA-256 of every file BEFORE unpickling, so a corrupted or modified model is
refused with a clear message. (Pickle-based formats can run code when loaded, so only ever load model
files you trained yourself — the checksum protects against accidents and tampering after training,
not against files from untrusted sources.)
"""

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import pandas as pd
import sklearn

from .exceptions import ModelError, NotTrainedError

MODEL_FILE, META_FILE, REFERENCE_FILE = "model.joblib", "metadata.json", "reference.csv"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic(path, writer):
    """Write to a temp file in the same folder, then rename — a crash never leaves a half-written file."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(fd)
    try:
        writer(tmp)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _json_default(value):
    if hasattr(value, "item"):          # numpy scalars
        return value.item()
    return str(value)


def save(result, directory):
    directory = Path(directory)
    try:
        directory.mkdir(parents=True, exist_ok=True)
        _atomic(directory / MODEL_FILE, lambda tmp: joblib.dump(result.pipeline, tmp, compress=3))
        _atomic(directory / REFERENCE_FILE, lambda tmp: result.reference.to_csv(tmp, index=False))
        metadata = dict(result.metadata)
        metadata["files"] = {MODEL_FILE: sha256(directory / MODEL_FILE),
                             REFERENCE_FILE: sha256(directory / REFERENCE_FILE)}

        def write_meta(tmp):
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(metadata, fh, indent=2, default=_json_default)
        _atomic(directory / META_FILE, write_meta)
    except OSError as exc:
        raise ModelError(f"Could not save the model to {directory}: {exc.strerror or exc}") from exc
    return directory


@dataclass
class LoadedModel:
    pipeline: object
    metadata: dict
    reference: pd.DataFrame
    directory: Path
    warnings: list = field(default_factory=list)


def is_trained(directory):
    return (Path(directory) / META_FILE).exists() and (Path(directory) / MODEL_FILE).exists()


def load(directory):
    directory = Path(directory)
    if not is_trained(directory):
        raise NotTrainedError(f"No trained model found in '{directory}'. Train one first "
                              "(menu option 'Train a model', or: python main.py --train).")
    try:
        metadata = json.loads((directory / META_FILE).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError) as exc:
        raise ModelError(f"Could not read the model's metadata: {exc}") from exc
    except json.JSONDecodeError:
        raise ModelError("The model's metadata file is damaged (not valid JSON). Train the model again.") from None
    files = metadata.get("files")
    if not isinstance(files, dict) or MODEL_FILE not in files:
        raise ModelError("The model's metadata has no checksums, so the model cannot be verified. "
                         "Train the model again.")
    for name, expected in files.items():
        target = directory / name
        if not target.exists():
            raise ModelError(f"Model file '{name}' is missing. Train the model again.")
        if sha256(target) != expected:
            raise ModelError(f"Model file '{name}' does not match its checksum: it is damaged or has been "
                             "modified. Train the model again.")
    warnings = []
    saved_version = metadata.get("environment", {}).get("sklearn")
    if saved_version and saved_version != sklearn.__version__:
        warnings.append(f"The model was trained with scikit-learn {saved_version} but {sklearn.__version__} is "
                        "installed; predictions may differ. Training again is recommended.")
    try:
        pipeline = joblib.load(directory / MODEL_FILE)
        reference = pd.read_csv(directory / REFERENCE_FILE)
    except Exception as exc:  # noqa: BLE001 - pickle can raise almost anything for incompatible files
        raise ModelError(f"The model could not be loaded ({type(exc).__name__}). Train it again with this "
                         "version of the libraries.") from None
    return LoadedModel(pipeline, metadata, reference, directory, warnings)
