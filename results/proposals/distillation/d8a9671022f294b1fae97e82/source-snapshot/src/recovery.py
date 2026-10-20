"""Explicit historical imports; missing SFT parents are never guessed or replaced."""

from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoTokenizer

from src.artifacts import file_hash
from src.training import _load_base, save_trained_model


def import_checkpoint(source, output, config, condition, sft_parent=None):
    source, output = Path(source), Path(output)
    if (condition == "sft_then_grpo") != bool(sft_parent):
        raise ValueError("Historical sft_then_grpo imports require the original SFT adapter explicitly")
    if output.exists():
        raise FileExistsError("Import destination must be new")
    model = _load_base(config["model"], config, device_map=None, dtype=torch.bfloat16)
    provenance = {
        "historical_condition": condition,
        "source_adapter_sha256": file_hash(source / "adapter_model.safetensors"),
        "chosen_base_revision": config["model"]["revision"],
        "historical_base_revision_verified": False,
        "evaluation_status": "pending",
    }
    if sft_parent:
        sft_parent = Path(sft_parent)
        provenance["sft_adapter_sha256"] = file_hash(sft_parent / "adapter_model.safetensors")
        model = PeftModel.from_pretrained(model, sft_parent).merge_and_unload(safe_merge=True)
    model = PeftModel.from_pretrained(model, source).merge_and_unload(safe_merge=True)
    # Use the pinned official tokenizer, rather than silently changing old metadata.
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["tokenizer"], revision=config["model"]["revision"])
    return save_trained_model(model, tokenizer, output, config, metadata={"historical_import": provenance})
