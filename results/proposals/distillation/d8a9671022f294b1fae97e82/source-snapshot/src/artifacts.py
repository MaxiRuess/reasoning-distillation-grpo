"""Immutable model identities and atomic, resumable evaluation transactions."""

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import uuid
from pathlib import Path

SCHEMA_VERSION = 1


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("w") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def runtime_metadata() -> dict:
    packages = {}
    for name in ["torch", "transformers", "trl", "peft", "datasets", "accelerate", "vllm", "math-verify"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True))
    except (OSError, subprocess.CalledProcessError):
        sha, dirty = None, None
    return {
        "python": platform.python_version(),
        "packages": packages,
        "code_sha": sha,
        "dirty": dirty,
        "source_digest": os.environ.get("SOURCE_DIGEST")
        or fingerprint({p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob("*.py"))}),
    }


def model_files(path: Path) -> dict:
    ignored = {"artifact.json", "run.json", "training_args.bin", "trainer_state.json", "data_manifest.json"}
    return {
        p.name: file_hash(p)
        for p in sorted(path.iterdir())
        if p.is_file() and p.name not in ignored and p.suffix in {".safetensors", ".bin", ".json", ".model", ".jinja"}
    }


def write_model_manifest(path, config: dict, artifact_format: str, parent: dict | None = None, metadata=None):
    path = Path(path)
    if artifact_format not in {"full", "adapter"}:
        raise ValueError("Unknown model artifact format")
    if (path / "artifact.json").exists():
        raise FileExistsError("Refusing to overwrite an existing model manifest")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "format": artifact_format,
        "base": config["model"],
        "parent": parent,
        "files": model_files(path),
        "metadata": metadata or {},
        "runtime": runtime_metadata(),
    }
    if not manifest["files"]:
        raise ValueError("Cannot manifest an empty artifact")
    manifest["artifact_id"] = fingerprint(manifest)
    atomic_json(path / "artifact.json", manifest)
    return manifest


def read_model_manifest(path, verify_files=True) -> dict:
    path = Path(path)
    with (path / "artifact.json").open() as stream:
        manifest = json.load(stream)
    identity = manifest.pop("artifact_id")
    valid = fingerprint(manifest) == identity and manifest["schema_version"] == SCHEMA_VERSION
    manifest["artifact_id"] = identity
    if not valid or (verify_files and manifest["files"] != model_files(path)):
        raise ValueError("Model artifact identity or files changed")
    if manifest["format"] not in {"full", "adapter"}:
        raise ValueError("Unknown model format")
    return manifest


def new_training_directory(root, condition: str, phase: str, config: dict) -> Path:
    if not condition.replace("_", "").isalnum() or phase not in {"sft", "grpo"}:
        raise ValueError("Invalid training condition or phase")
    run = fingerprint(config)[:12] + "-" + uuid.uuid4().hex[:12]
    path = Path(root) / condition / run / phase
    path.mkdir(parents=True, exist_ok=False)
    atomic_json(path / "source.json", {p.name: p.read_text() for p in sorted(Path(__file__).parent.glob("*.py"))})
    atomic_json(path / "run.json", {"config": config, "runtime": runtime_metadata()})
    return path


class EvaluationStore:
    """One fsync'd JSONL transaction per problem; repair only an interrupted final write."""

    def __init__(self, path: Path, identity: dict, problem_ids: list[str], num_samples: int):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        if len(set(problem_ids)) != len(problem_ids) or num_samples < 1:
            raise ValueError("Problem IDs must be unique and sample count positive")
        self.problem_ids, self.num_samples = problem_ids, num_samples
        expected = {
            "schema_version": SCHEMA_VERSION,
            "identity": identity,
            "problem_ids": problem_ids,
            "num_samples": num_samples,
        }
        manifest_path = self.path / "evaluation.json"
        if manifest_path.exists():
            if json.loads(manifest_path.read_text()) != expected:
                raise ValueError("Resume identity mismatch; start a separate evaluation run")
        else:
            if (self.path / "predictions.jsonl").exists():
                raise ValueError("Predictions without an evaluation manifest")
            atomic_json(manifest_path, expected)
        self.records = []
        predictions = self.path / "predictions.jsonl"
        if predictions.exists():
            raw = predictions.read_bytes()
            valid = 0
            for line in raw.splitlines(keepends=True):
                if not line.endswith(b"\n"):
                    # A transaction is committed only by its final newline.
                    break
                record = json.loads(line)
                self.records.append(record)
                valid += len(line)
            complete_count = len(self.records) // num_samples * num_samples
            if complete_count != len(self.records):
                self.records = self.records[:complete_count]
                valid = sum(len(line) for line in raw.splitlines(keepends=True)[:complete_count])
            if valid != len(raw):
                with predictions.open("r+b") as stream:
                    stream.truncate(valid)
            self._validate(self.records)

    @property
    def completed_problems(self):
        return len(self.records) // self.num_samples

    def _validate(self, records):
        if len(records) % self.num_samples or len(records) > len(self.problem_ids) * self.num_samples:
            raise ValueError("Incomplete or extra evaluation records")
        for index, record in enumerate(records):
            if (
                record["problem_id"] != self.problem_ids[index // self.num_samples]
                or record["sample_id"] != index % self.num_samples
            ):
                raise ValueError("Duplicate, missing, or reordered problem/sample ID")

    def append(self, records: list[dict]):
        self._validate(self.records + records)
        payload = "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in records)
        with (self.path / "predictions.jsonl").open("a") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        self.records.extend(records)
