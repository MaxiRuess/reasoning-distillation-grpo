"""Warm pinned model and benchmark files on a CPU container, before GPU allocation."""

import os
import time
from pathlib import Path

from huggingface_hub import snapshot_download

from src.artifacts import atomic_json, file_hash
from src.data import load_benchmark


def prepare_runtime_cache(config):
    start = time.monotonic()
    snapshot = Path(
        snapshot_download(
            config["model"]["name"],
            revision=config["model"]["revision"],
            allow_patterns=[
                "*.safetensors",
                "*.safetensors.index.json",
                "config.json",
                "generation_config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "*.jinja",
                "vocab.json",
                "merges.txt",
                "added_tokens.json",
                "tokenizer.model",
            ],
        )
    )
    weights = sorted(snapshot.glob("*.safetensors"))
    if not weights:
        raise ValueError("Pinned model cache has no safetensors weights")
    benchmarks = {name: load_benchmark(name, config)._fingerprint for name in config["evaluation"]["benchmarks"]}
    manifest = {
        "model": config["model"],
        "snapshot": str(snapshot),
        "weights_sha256": {p.name: file_hash(p) for p in weights},
        "benchmark_fingerprints": benchmarks,
        "elapsed_seconds": time.monotonic() - start,
    }
    atomic_json(Path(os.environ.get("HF_HOME", "data/hf-cache")) / "runtime-cache.json", manifest)
    return manifest
