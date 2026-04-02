"""Model loading, LoRA setup, and trainer construction for SFT and GRPO."""

import torch
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import GRPOConfig, GRPOTrainer, SFTConfig, SFTTrainer


def load_model(config: dict) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load the base model, optionally with 4-bit quantization.

    When config["quantization"]["enabled"] is false, loads in full bfloat16.
    Otherwise applies NF4 quantization via bitsandbytes.

    Returns (model, tokenizer) with tokenizer configured for training
    (right-padded). Switch to left-padding before generation.
    """
    quant_cfg = config["quantization"]
    use_quantization = quant_cfg.get("enabled", quant_cfg.get("load_in_4bit", False))

    if use_quantization:
        compute_dtype = getattr(torch, quant_cfg["bnb_4bit_compute_dtype"])
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=quant_cfg["bnb_4bit_quant_type"],
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=quant_cfg["bnb_4bit_use_double_quant"],
        )
        model = AutoModelForCausalLM.from_pretrained(
            config["model"]["name"],
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=compute_dtype,
            attn_implementation="flash_attention_2",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            config["model"]["name"],
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )

    tokenizer = AutoTokenizer.from_pretrained(config["model"]["tokenizer"])

    # Qwen3 base model uses <|endoftext|> as eos_token by default, but the
    # chat template uses <|im_end|> as the end-of-turn marker. Align them
    # so the model learns to stop generating at the right token.
    eos_token = config["model"].get("eos_token")
    if eos_token:
        tokenizer.eos_token = eos_token

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer


def get_lora_config(config: dict) -> LoraConfig:
    """Build LoraConfig from the config dictionary.

    Supports rsLoRA via use_rslora flag — scales the LoRA adapter by
    alpha/sqrt(r) instead of alpha/r, preventing gradient collapse at
    high ranks (arXiv:2312.03732).
    """
    lora_cfg = config["lora"]
    return LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        bias=lora_cfg["bias"],
        task_type=lora_cfg["task_type"],
        target_modules=lora_cfg["target_modules"],
        use_rslora=lora_cfg.get("use_rslora", False),
    )


def build_sft_trainer(
    model,
    tokenizer,
    dataset,
    condition_config: dict,
    lora_config: LoraConfig,
) -> SFTTrainer:
    """Construct an SFTTrainer from condition-specific config.

    TRL handles prepare_model_for_kbit_training and get_peft_model
    internally when peft_config is provided.
    """
    sft_cfg = condition_config["sft"]

    training_args = SFTConfig(
        output_dir=sft_cfg["output_dir"],
        learning_rate=sft_cfg["learning_rate"],
        num_train_epochs=sft_cfg["num_train_epochs"],
        per_device_train_batch_size=sft_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=sft_cfg["gradient_accumulation_steps"],
        warmup_steps=int(sft_cfg.get("warmup_steps", 10)),
        optim=sft_cfg["optim"],
        max_length=sft_cfg["max_seq_length"],
        packing=sft_cfg.get("packing", False),
        use_liger_kernel=sft_cfg.get("use_liger_kernel", False),
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to="wandb",
        gradient_checkpointing=True,
    )

    return SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        peft_config=lora_config,
    )


def build_grpo_trainer(
    model,
    tokenizer,
    dataset,
    condition_config: dict,
    lora_config: LoraConfig,
    reward_fns: list,
) -> GRPOTrainer:
    """Construct a GRPOTrainer from condition-specific config.

    The reward_fns list should contain callable reward functions that
    accept (completions, answer, **kwargs) and return list[float].
    """
    grpo_cfg = condition_config["grpo"]

    training_args = GRPOConfig(
        output_dir=grpo_cfg["output_dir"],
        learning_rate=grpo_cfg["learning_rate"],
        num_train_epochs=grpo_cfg["num_train_epochs"],
        per_device_train_batch_size=grpo_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=grpo_cfg["gradient_accumulation_steps"],
        num_generations=grpo_cfg["num_generations"],
        max_completion_length=grpo_cfg["max_completion_length"],
        optim=grpo_cfg["optim"],
        use_liger_kernel=grpo_cfg.get("use_liger_kernel", False),
        bf16=True,
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        report_to="wandb",
        gradient_checkpointing=True,
    )

    return GRPOTrainer(
        model=model,
        reward_funcs=reward_fns,
        args=training_args,
        train_dataset=dataset,
        peft_config=lora_config,
    )


def load_sft_checkpoint(
    checkpoint_path: str, config: dict
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load a model from an SFT checkpoint for the GRPO phase.

    Loads the base model (quantized or full bf16 based on config), then
    applies the LoRA adapter from the checkpoint and merges it into the
    base weights so GRPO can apply a fresh adapter on top.
    """
    quant_cfg = config["quantization"]
    use_quantization = quant_cfg.get("enabled", quant_cfg.get("load_in_4bit", False))

    if use_quantization:
        compute_dtype = getattr(torch, quant_cfg["bnb_4bit_compute_dtype"])
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=quant_cfg["bnb_4bit_quant_type"],
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=quant_cfg["bnb_4bit_use_double_quant"],
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            config["model"]["name"],
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=compute_dtype,
            attn_implementation="flash_attention_2",
        )
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            config["model"]["name"],
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )

    # Load the LoRA adapter from the SFT checkpoint
    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    # Merge and unload so GRPO can apply a fresh adapter
    model = model.merge_and_unload()

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

    # Align EOS token with chat template (same as load_model)
    eos_token = config["model"].get("eos_token")
    if eos_token:
        tokenizer.eos_token = eos_token

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer
