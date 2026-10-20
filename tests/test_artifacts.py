import tempfile
import unittest
from pathlib import Path

from src.artifacts import EvaluationStore, fingerprint, read_model_manifest, write_model_manifest


class ArtifactTests(unittest.TestCase):
    def test_modified_weights_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / "model.safetensors").write_bytes(b"weights")
            write_model_manifest(path, {"model": {"name": "model", "revision": "a" * 40}}, "full")
            self.assertEqual(read_model_manifest(path)["format"], "full")
            (path / "model.safetensors").write_bytes(b"changed")
            with self.assertRaises(ValueError):
                read_model_manifest(path)

    def test_resume_identity_and_unique_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            store = EvaluationStore(path, {"model": "one", "tokens": 10}, ["q1", "q2"], 1)
            store.append([{"problem_id": "q1", "sample_id": 0, "ground_truth": "2", "response": "2"}])
            resumed = EvaluationStore(path, {"model": "one", "tokens": 10}, ["q1", "q2"], 1)
            self.assertEqual(resumed.completed_problems, 1)
            with self.assertRaises(ValueError):
                EvaluationStore(path, {"model": "two", "tokens": 10}, ["q1", "q2"], 1)
            with self.assertRaises(ValueError):
                resumed.append([{"problem_id": "q1", "sample_id": 0}])

    def test_truncated_last_transaction_is_recoverable(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = EvaluationStore(Path(tmp), {"model": "one"}, ["q"], 2)
            with open(store.path / "predictions.jsonl", "a") as f:
                f.write('{"broken":')
            resumed = EvaluationStore(Path(tmp), {"model": "one"}, ["q"], 2)
            self.assertEqual(resumed.records, [])
            resumed.append([{"problem_id": "q", "sample_id": i} for i in range(2)])
            self.assertEqual(len(EvaluationStore(Path(tmp), {"model": "one"}, ["q"], 2).records), 2)

    def test_fingerprint_is_order_independent(self):
        self.assertEqual(fingerprint({"a": 1, "b": 2}), fingerprint({"b": 2, "a": 1}))
