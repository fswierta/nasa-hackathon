"""Query encoder without chromadb (issue #51): all-MiniLM-L6-v2 ONNX via onnxruntime + tokenizers.

Same model file, tokenizer and maths as chromadb's DefaultEmbeddingFunction (ONNXMiniLM_L6_V2):
truncate at 256 tokens, run the ONNX graph, mean-pool the last hidden state over the attention mask,
L2-normalise. Padding is to the longest text in the batch instead of always 256 — masked positions don't
change the result, it is only faster (equivalence is tested: cosine >= 0.999 vs chromadb).

The model (~80 MB archive) is downloaded once to the same folder chromadb uses
(~/.cache/chroma/onnx_models/all-MiniLM-L6-v2), so an existing download is reused. Offline / blocked
download -> EncoderUnavailable with a clear message; ai.vectors then reports it and search uses keywords.
"""
from __future__ import annotations

import hashlib
import os
import tarfile
import urllib.request
from functools import lru_cache
from pathlib import Path

import numpy as np

MODEL_NAME = "all-MiniLM-L6-v2"
MODEL_DIR = Path(os.environ.get("FIF_ENCODER_DIR") or Path.home() / ".cache" / "chroma" / "onnx_models" / MODEL_NAME)
MODEL_URL = "https://chroma-onnx-models.s3.amazonaws.com/all-MiniLM-L6-v2/onnx.tar.gz"
MODEL_SHA256 = "913d7300ceae3b2dbc2c50d1de4baacab4be7b9380491c27fab7418616a16ec3"  # same check as chromadb
MAX_TOKENS = 256
BATCH = 32


class EncoderUnavailable(RuntimeError):
    """The model could not be loaded (offline, blocked download, missing package)."""


def _files_present() -> bool:
    d = MODEL_DIR / "onnx"
    return (d / "model.onnx").exists() and (d / "tokenizer.json").exists()


def ensure_model() -> Path:
    """Model folder, downloading + verifying + extracting the archive once if needed."""
    if _files_present():
        return MODEL_DIR / "onnx"
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    archive = MODEL_DIR / "onnx.tar.gz"
    try:
        if not archive.exists():
            tmp = archive.with_suffix(".part")
            with urllib.request.urlopen(MODEL_URL, timeout=60) as r, open(tmp, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            tmp.replace(archive)
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        if digest != MODEL_SHA256:
            archive.unlink(missing_ok=True)
            raise EncoderUnavailable("downloaded model archive failed its SHA-256 check (deleted; retry)")
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(MODEL_DIR, filter="data")
    except EncoderUnavailable:
        raise
    except Exception as e:
        raise EncoderUnavailable(f"could not download the {MODEL_NAME} model ({e.__class__.__name__}: {e}); "
                                 "search falls back to keywords") from e
    if not _files_present():
        raise EncoderUnavailable("model archive did not contain onnx/model.onnx and tokenizer.json")
    return MODEL_DIR / "onnx"


@lru_cache(maxsize=1)
def _session():
    try:
        import onnxruntime as ort
        from tokenizers import Tokenizer
    except ImportError as e:
        raise EncoderUnavailable(f"missing package: {e.name} (pip install onnxruntime tokenizers)") from e
    d = ensure_model()
    tok = Tokenizer.from_file(str(d / "tokenizer.json"))
    tok.enable_truncation(max_length=MAX_TOKENS)
    tok.enable_padding(pad_id=0, pad_token="[PAD]")  # pad to the longest in the batch
    so = ort.SessionOptions()
    so.log_severity_level = 3
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads = min(4, os.cpu_count() or 1)
    sess = ort.InferenceSession(str(d / "model.onnx"), providers=["CPUExecutionProvider"], sess_options=so)
    return tok, sess


def embed(texts: list[str]) -> np.ndarray:
    """L2-normalised 384-d embeddings, float32, one row per text."""
    tok, sess = _session()
    out = []
    for i in range(0, len(texts), BATCH):
        enc = tok.encode_batch(list(texts[i:i + BATCH]))
        ids = np.array([e.ids for e in enc], dtype=np.int64)
        mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
        hidden = sess.run(None, {"input_ids": ids, "attention_mask": mask,
                                 "token_type_ids": np.zeros_like(ids)})[0]
        m = mask[..., None].astype(np.float32)
        pooled = (hidden * m).sum(axis=1) / np.clip(m.sum(axis=1), 1e-9, None)
        norm = np.linalg.norm(pooled, axis=1, keepdims=True)
        out.append((pooled / np.where(norm == 0, 1e-12, norm)).astype(np.float32))
    return np.vstack(out) if out else np.zeros((0, 384), dtype=np.float32)
