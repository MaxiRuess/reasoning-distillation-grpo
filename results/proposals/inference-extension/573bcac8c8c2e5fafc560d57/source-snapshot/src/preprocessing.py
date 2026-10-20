"""CPU-only preparation of the exact packed inputs consumed by the pinned SFTTrainer."""

import importlib.metadata
import inspect
import json
import tempfile
from pathlib import Path

from datasets import load_from_disk

from src.artifacts import atomic_json, file_hash, fingerprint


def tokenization_identity(tokenizer, raw, cfg):
    from trl import SFTTrainer

    return {
        "raw_content_hash": raw["content_hash"],
        "max_length": cfg["max_seq_length"],
        "packing": cfg.get("packing", False),
        "packing_strategy": "bfd",
        "completion_only_loss": True,
        "template": tokenizer.chat_template,
        "tokenizer_hash": fingerprint(tokenizer.backend_tokenizer.to_str()),
        "eos_token": tokenizer.eos_token,
        "pad_token": tokenizer.pad_token,
        "versions": {name: importlib.metadata.version(name) for name in ["trl", "transformers", "tokenizers"]},
        "trainer_source": file_hash(Path(inspect.getfile(SFTTrainer))),
        "preprocessor_source": file_hash(Path(__file__)),
    }


def validate_sft_rows(dataset, tokenizer, maximum):
    for row in dataset:
        ids = row["input_ids"]
        lengths = row.get("seq_lengths", [len(ids)])
        if not ids or len(ids) > maximum or sum(lengths) != len(ids):
            raise ValueError("Invalid processed sequence length or packing boundaries")
        mask = row.get("completion_mask")
        if mask is None or len(mask) != len(ids) or not any(mask):
            raise ValueError("Processed SFT target has no aligned completion supervision")
        offset = 0
        for length in lengths:
            segment = ids[offset : offset + length]
            if not tokenizer.decode(segment, skip_special_tokens=False).rstrip().endswith(tokenizer.eos_token):
                raise ValueError("Processed SFT sequence lost its final EOS")
            offset += length


def prepare_sft_tokens(dataset, tokenizer, raw, cfg):
    """Use real TRL preprocessing with a ~5 MB CPU model; never load pretrained weights."""
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from trl import SFTConfig, SFTTrainer

    identity = tokenization_identity(tokenizer, raw, cfg)
    path = Path(raw["path"]) / "sft-tokens" / fingerprint(identity)[:24]
    if (path / "tokens.json").exists():
        return load_sft_tokens(tokenizer, raw, cfg)[1]
    vocab = max(len(tokenizer), max(tokenizer.get_vocab().values()) + 1)
    model = Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=vocab,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=1,
            num_key_value_heads=1,
            head_dim=8,
            tie_word_embeddings=True,
        )
    )
    with tempfile.TemporaryDirectory() as output:
        args = SFTConfig(
            output_dir=output,
            use_cpu=True,
            bf16=False,
            fp16=False,
            report_to="none",
            max_length=cfg["max_seq_length"],
            packing=cfg.get("packing", False),
            packing_strategy="bfd",
            completion_only_loss=True,
            eos_token=tokenizer.eos_token,
            shuffle_dataset=False,
            gradient_checkpointing=False,
            use_liger_kernel=False,
            save_strategy="no",
            seed=2026,
        )
        trainer = SFTTrainer(model=model, args=args, train_dataset=dataset, processing_class=tokenizer)
        prepared = trainer.train_dataset.select_columns(
            [
                key
                for key in ["input_ids", "completion_mask", "seq_lengths"]
                if key in trainer.train_dataset.column_names
            ]
        )
        validate_sft_rows(prepared, tokenizer, cfg["max_seq_length"])
        path.mkdir(parents=True, exist_ok=False)
        prepared.save_to_disk(str(path / "dataset"))
    files = {file.name: file_hash(file) for file in (path / "dataset").iterdir() if file.is_file()}
    meta = {
        "schema_version": 1,
        "identity": identity,
        "path": str(path),
        "rows": len(prepared),
        "files": files,
        "processed_tokens": sum(len(row["input_ids"]) for row in prepared),
        "supervised_tokens": sum(sum(row["completion_mask"]) for row in prepared),
    }
    atomic_json(path / "tokens.json", meta)
    return meta


def load_sft_tokens(tokenizer, raw, cfg):
    identity = tokenization_identity(tokenizer, raw, cfg)
    path = Path(raw["path"]) / "sft-tokens" / fingerprint(identity)[:24]
    if not (path / "tokens.json").exists():
        raise FileNotFoundError("Tokenized SFT inputs missing; prepare the condition on CPU before allocating a GPU")
    meta = json.loads((path / "tokens.json").read_text())
    actual = {file.name: file_hash(file) for file in (path / "dataset").iterdir() if file.is_file()}
    if meta["identity"] != identity or meta["files"] != actual:
        raise ValueError("Prepared SFT token files changed")
    return load_from_disk(str(path / "dataset")), meta
