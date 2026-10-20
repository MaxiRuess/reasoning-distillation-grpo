"""Modal entry points share exactly the same training/evaluation paths as local CLIs.

Importing this module or running the pipeline dry-run never starts a GPU Function.
"""

import json
from pathlib import Path

import modal

from src.artifacts import file_hash, fingerprint

app = modal.App("reasoning-distillation-grpo")
SOURCE_DIGEST = fingerprint(
    {
        str(path): file_hash(path)
        for path in [*sorted(Path("src").glob("*.py")), Path("modal_train.py"), Path("configs/config.yaml")]
    }
)
FLASH_ATTN_WHEEL = (
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/"
    "flash_attn-2.8.3+cu12torch2.8cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
)


def image_for(lock, flash=False, offline=False):
    image = modal.Image.debian_slim(python_version="3.11").pip_install_from_requirements(lock)
    if flash:
        image = image.run_commands(
            "python -c "
            + __import__("shlex").quote(
                "import subprocess,sys,torch; abi=str(torch.compiled_with_cxx11_abi()).upper(); "
                "url=" + repr(FLASH_ATTN_WHEEL) + ".replace('abiFALSE','abi'+abi); "
                "subprocess.check_call([sys.executable,'-m','pip','install',url])"
            )
        )
    environment = {"PYTHONPATH": "/root", "SOURCE_DIGEST": SOURCE_DIGEST, "HF_HOME": "/vol/hf-cache-v2"}
    if offline:
        environment.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1")
    return (
        image.env(environment)
        .add_local_dir("src", remote_path="/root/src")
        .add_local_file("configs/config.yaml", remote_path="/root/configs/config.yaml")
    )


sft_image = image_for("requirements/sft.lock", flash=True)
grpo_image = image_for("requirements/grpo.lock")
sft_gpu_image = image_for("requirements/sft.lock", flash=True, offline=True)
grpo_gpu_image = image_for("requirements/grpo.lock", offline=True)
volume = modal.Volume.from_name("reasoning-distillation-vol", create_if_missing=False)
VOLUME_PATH = "/vol"
SECRETS = [modal.Secret.from_name("wandb-secret"), modal.Secret.from_name("huggingface-secret")]


@app.function(image=sft_image, timeout=3600, secrets=SECRETS, volumes={VOLUME_PATH: volume})
def prepare_runtime():
    from src.data import load_config
    from src.runtime_cache import prepare_runtime_cache

    volume.reload()
    result = prepare_runtime_cache(load_config("/root/configs/config.yaml"))
    volume.commit()
    return result


@app.function(image=sft_image, timeout=6 * 3600, secrets=SECRETS, volumes={VOLUME_PATH: volume})
def prepare_data(condition: str, phase: str = "sft"):
    from src.data import load_config
    from src.pipeline import prepare_phase

    volume.reload()
    result = prepare_phase(load_config("/root/configs/config.yaml"), condition, phase, "/vol/prepared-v2")
    volume.commit()
    return {"rows": result["rows"], "path": result["manifest"]["path"]}


def _distillation_config(protocol):
    import os

    if fingerprint({k: v for k, v in protocol.items() if k != "protocol_id"}) != protocol["protocol_id"]:
        raise ValueError("Distillation protocol was modified")
    bundled = {
        k: v
        for k, v in protocol["implementation"].items()
        if k.startswith("src/") or k in {"modal_train.py", "configs/config.yaml"}
    }
    if os.environ["SOURCE_DIGEST"] != fingerprint(bundled):
        raise ValueError("Bundled source differs from the frozen distillation protocol")
    return protocol["config"]


@app.function(image=sft_image, timeout=900, secrets=SECRETS, volumes={VOLUME_PATH: volume})
def prepare_distillation_data(condition: str, protocol: dict, phase: str = "sft"):
    from src.pipeline import prepare_phase

    volume.reload()
    config = _distillation_config(protocol)
    try:
        result = prepare_phase(config, condition, phase, "/vol/distillation-prepared")
        expected = protocol["prepared"][condition]
        if (
            result["manifest"]["content_hash"] != expected["content_hash"]
            or result["manifest"]["source_ids"] != expected["source_ids"]
        ):
            raise ValueError("Remote training data differs from the frozen local preparation")
        for key in ("rows", "processed_tokens", "supervised_tokens"):
            if result["tokenized_manifest"][key] != expected["tokens"][key]:
                raise ValueError("Remote tokenized data differs from frozen exposure counts")
        return {"rows": result["rows"], "content_hash": result["manifest"]["content_hash"]}
    finally:
        volume.commit()


@app.function(image=sft_gpu_image, gpu="L40S", timeout=2400, secrets=SECRETS, volumes={VOLUME_PATH: volume})
def train_distillation(condition: str, seed: int, protocol: dict, pilot: bool = False):
    from copy import deepcopy

    from src.pipeline import train_phase

    volume.reload()
    config = deepcopy(_distillation_config(protocol))
    if pilot or seed not in (42, 43, 44) or condition not in config["conditions"]:
        raise ValueError("Training request is outside the frozen study")
    config["seed"] = seed
    try:
        return train_phase(
            config, condition, "sft", output_root="/vol/distillation-training", data_root="/vol/distillation-prepared"
        )
    finally:
        volume.commit()


@app.function(image=sft_gpu_image, gpu="L40S", timeout=6 * 3600, secrets=SECRETS, volumes={VOLUME_PATH: volume})
def _train_sft_job(condition: str = "sft_traces", seed: int = 42, pilot: bool = False):
    from src.data import load_config
    from src.pipeline import train_phase

    volume.reload()
    config = load_config("/root/configs/config.yaml")
    config["seed"] = seed
    if pilot:
        phase = "sft"
        config["conditions"][condition][phase].update(
            max_steps=config["performance"]["pilot_" + phase + "_steps"], logging_steps=1, save_strategy="no"
        )
        config["pilot"] = True
    try:
        return train_phase(config, condition, "sft", output_root="/vol/training-v2", data_root="/vol/prepared-v2")
    finally:
        volume.commit()


@app.function(image=grpo_gpu_image, gpu="H100", timeout=12 * 3600, secrets=SECRETS, volumes={VOLUME_PATH: volume})
def _train_grpo_job(
    condition: str = "grpo_only", sft_checkpoint: str | None = None, seed: int = 42, pilot: bool = False
):
    import os

    os.environ.update(RANK="0", LOCAL_RANK="0", WORLD_SIZE="1", MASTER_ADDR="localhost", MASTER_PORT="12355")
    from src.data import load_config
    from src.pipeline import train_phase

    volume.reload()
    config = load_config("/root/configs/config.yaml")
    config["seed"] = seed
    if pilot:
        phase = "grpo"
        config["conditions"][condition][phase].update(
            max_steps=config["performance"]["pilot_" + phase + "_steps"], logging_steps=1, save_strategy="no"
        )
        config["pilot"] = True
    try:
        return train_phase(config, condition, "grpo", sft_checkpoint, "/vol/training-v2", "/vol/prepared-v2")
    finally:
        try:
            volume.commit()
        finally:
            from torch import distributed

            if distributed.is_initialized():
                distributed.destroy_process_group()


@app.function(image=sft_gpu_image, gpu="L40S", timeout=12 * 3600, secrets=SECRETS, volumes={VOLUME_PATH: volume})
def run_evaluation(
    model_path: str,
    condition: str,
    benchmark: str = "gsm8k",
    max_new_tokens: int = 2048,
    num_samples: int = 1,
    temperature: float = 0.0,
    seed: int = 42,
    pilot: bool = False,
    problem_indices: list[int] | None = None,
    expected_rows_digest: str | None = None,
):
    from src.data import load_config
    from src.evaluation import run_all_evaluations

    volume.reload()
    config = load_config("/root/configs/config.yaml")
    config["evaluation"].update(
        benchmarks=[benchmark],
        max_new_tokens=max_new_tokens,
        num_samples_per_problem=num_samples,
        temperature=temperature,
        seed=seed,
        output_root="/vol/evaluations-v2",
    )
    if pilot:
        config["evaluation"]["max_problems"] = config["performance"]["pilot_eval_problems"]
        config["evaluation"]["telemetry_interval_seconds"] = 1
    if problem_indices is not None:
        config["evaluation"].update(problem_indices=problem_indices, expected_rows_digest=expected_rows_digest)
    return run_all_evaluations(model_path, config, condition, on_batch=volume.commit)


@app.function(image=sft_gpu_image, timeout=1800, secrets=SECRETS, volumes={VOLUME_PATH: volume})
def reconstruct_checkpoint(condition: str, protocol: dict, parent_path: str | None = None):
    import os

    from src.checkpoint_experiment import immutable_json, reconstruct_arm
    from src.data import load_config

    expected = fingerprint({k: v for k, v in protocol["implementation"].items() if not k.startswith("scripts/")})
    if os.environ["SOURCE_DIGEST"] != expected:
        raise ValueError("Bundled reconstruction code differs from the frozen protocol")
    volume.reload()
    root = Path("/vol/checkpoint-ablation") / protocol["protocol_id"][:24]
    immutable_json(root / "protocol.json", protocol)
    try:
        return reconstruct_arm(
            condition, protocol, load_config("/root/configs/config.yaml"), root / "models", parent_path=parent_path
        )
    finally:
        volume.commit()


@app.function(image=sft_gpu_image, gpu="L40S", timeout=12 * 3600, secrets=SECRETS, volumes={VOLUME_PATH: volume})
def full_eval_gsm8k(
    model_path: str = "/vol/outputs/sft_traces", condition: str = "sft_traces", base_model_only: bool = False
):
    from src.data import load_config
    from src.evaluation import run_all_evaluations

    volume.reload()
    config = load_config("/root/configs/config.yaml")
    config["evaluation"].update(benchmarks=["gsm8k"], output_root="/vol/evaluations-v2")
    path = config["model"]["name"] if base_model_only else model_path
    return run_all_evaluations(path, config, condition, on_batch=volume.commit)


@app.function(image=sft_image, timeout=300, secrets=SECRETS, volumes={VOLUME_PATH: volume})
def verify_volume_result(result_path: str):
    """Separate-container read verifies a committed result outlives its producer."""
    from src.artifacts import file_hash

    volume.reload()
    path = Path(result_path)
    if not path.is_relative_to("/vol/evaluations-v2"):
        raise ValueError("Expected a v2 result on the mounted volume")
    return {
        "summary": json.loads((path / "summary.json").read_text()),
        "predictions_sha256": file_hash(path / "predictions.jsonl"),
    }


@app.local_entrypoint()
def run_full_pipeline(experiment: str = "controlled", dry_run: bool = True, quote: str = "", budget_usd: float = 0):
    from src.data import load_config
    from src.study import build_pilot_plan, build_study_plan, execute_study

    config = load_config()
    plan = build_pilot_plan(config) if experiment == "pilot" else build_study_plan(config, experiment)
    if dry_run:
        print(json.dumps(plan, indent=2))
        return

    def limited(function):
        def call(_timeout, **kwargs):
            return function.with_options(
                timeout=_timeout, scaledown_window=2, max_containers=1, retries=0, cpu=(2, 2), memory=(16384, 16384)
            ).remote(**kwargs)

        return call

    execute_study(
        plan,
        config,
        quote,
        budget_usd,
        prepare=limited(prepare_data),
        sft=limited(_train_sft_job),
        grpo=limited(_train_grpo_job),
        evaluate=limited(run_evaluation),
        verify=limited(verify_volume_result),
        prepare_runtime=limited(prepare_runtime),
    )


@app.local_entrypoint()
def train_sft(condition: str = "sft_traces", seed: int = 42, pilot: bool = False):
    from src.data import load_config
    from src.pipeline import validate_training_request

    config = load_config()
    config["seed"] = seed
    validate_training_request(config, condition, "sft")
    print(_train_sft_job.remote(condition=condition, seed=seed, pilot=pilot))


@app.local_entrypoint()
def train_grpo(condition: str = "grpo_only", sft_checkpoint: str | None = None, seed: int = 42, pilot: bool = False):
    from src.data import load_config
    from src.pipeline import validate_training_request

    config = load_config()
    config["seed"] = seed
    validate_training_request(config, condition, "grpo", sft_checkpoint, check_artifact=False)
    print(_train_grpo_job.remote(condition=condition, sft_checkpoint=sft_checkpoint, seed=seed, pilot=pilot))
