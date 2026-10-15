"""BlueLM-7B-Chat inference and LoRA entry point for a MUSA device.

The tutorial under ``models/BlueLM`` is kept unchanged.  This adapter loads a
local Transformers checkpoint explicitly on MUSA and supports optional PEFT
adapter loading or a small JSON LoRA training run.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("infer", "lora"))
    parser.add_argument(
        "--model-dir", "--model-path", "--base-model", dest="model_dir", required=True,
        type=Path, help="本地 BlueLM 基础模型目录",
    )
    parser.add_argument("--adapter-path", type=Path, help="可选的 PEFT LoRA 目录")
    parser.add_argument("--prompt", default="请介绍一下你自己。")
    parser.add_argument("--device", default="musa:0", help="Torch device，例如 musa:0")
    parser.add_argument("--data-json", type=Path, help="LoRA 数据 JSON 数组")
    parser.add_argument("--output-dir", type=Path, help="LoRA 输出目录")
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    return parser.parse_args()


def _load_runtime(device_name: str) -> tuple[Any, Any, Any, Any]:
    if device_name.startswith("musa"):
        os.environ.setdefault("TORCHADA_FORCE_CUDA_AVAILABLE", "1")
        try:
            import torchada  # noqa: F401  # enables CUDA-compatible MUSA checks
            import torch_musa  # noqa: F401  # registers the MUSA backend
        except ImportError as exc:
            raise RuntimeError("MUSA inference requires torch_musa") from exc
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    return torch, AutoModelForCausalLM, AutoTokenizer, device_name


def _prompt_text(tokenizer: Any, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except (AttributeError, TypeError, ValueError):
            pass
    return f"[|Human|]:{prompt}\n[|AI|]:"


def _load_model(torch: Any, model_cls: Any, tokenizer_cls: Any, args: argparse.Namespace) -> tuple[Any, Any]:
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    tokenizer = tokenizer_cls.from_pretrained(
        str(args.model_dir), local_files_only=True, use_fast=False, trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model = model_cls.from_pretrained(
        str(args.model_dir), local_files_only=True, torch_dtype=dtype, trust_remote_code=True,
    ).to(args.device)
    if args.device.startswith("musa") and any(
        p.device.type != "musa" or p.device.index != 0 for p in model.parameters()
    ):
        raise RuntimeError("模型参数未全部位于 MUSA device 0")
    if args.adapter_path is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(args.adapter_path)).to(args.device)
    return tokenizer, model.eval()


def _records(args: argparse.Namespace, tokenizer: Any) -> list[dict[str, list[int]]]:
    if args.data_json is None:
        raise ValueError("lora 需要 --data-json")
    rows = json.loads(args.data_json.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("--data-json 必须是非空数组")
    result = []
    for row in rows:
        prompt = str(row["instruction"]) + str(row.get("input", ""))
        prompt_ids = tokenizer(_prompt_text(tokenizer, prompt), add_special_tokens=False)["input_ids"]
        answer_ids = tokenizer(str(row["output"]), add_special_tokens=False)["input_ids"]
        if tokenizer.eos_token_id is not None:
            answer_ids.append(tokenizer.eos_token_id)
        ids = (prompt_ids + answer_ids)[: args.max_length]
        labels = ([-100] * len(prompt_ids) + answer_ids)[: args.max_length]
        if not any(value != -100 for value in labels):
            raise ValueError("--max-length 太小，未保留答案 token")
        result.append({"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels})
    return result


def _run_infer(torch: Any, tokenizer: Any, model: Any, args: argparse.Namespace) -> None:
    inputs = tokenizer(_prompt_text(tokenizer, args.prompt), return_tensors="pt").to(args.device)
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
    if args.device.startswith("musa") and output.device.type != "musa":
        raise RuntimeError("生成结果未位于 MUSA")
    answer = tokenizer.decode(output[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
    print(answer)
    print(json.dumps({"action": "infer", "status": "PASS", "device": str(output.device)}))


def _run_lora(tokenizer: Any, model: Any, args: argparse.Namespace) -> None:
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments

    if args.output_dir is None:
        raise ValueError("lora 需要 --output-dir")
    examples = _records(args, tokenizer)
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        r=8, lora_alpha=32, lora_dropout=0.1,
    ))
    model.enable_input_require_grads()
    model.config.use_cache = False
    train_args = TrainingArguments(
        output_dir=str(args.output_dir), per_device_train_batch_size=1, max_steps=args.max_steps,
        learning_rate=1e-4, logging_steps=1, save_strategy="no", report_to="none",
        bf16=args.dtype == "bfloat16", fp16=False, gradient_checkpointing=True, dataloader_num_workers=0,
    )
    trainer = Trainer(model=model, args=train_args, train_dataset=examples,
                      data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True))
    result = trainer.train()
    model.save_pretrained(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    print(json.dumps({"action": "lora", "status": "PASS", "device": str(next(model.parameters()).device),
                      "steps": result.global_step, "adapter_directory": str(args.output_dir)}))


def main() -> None:
    args = _parse_args()
    if not args.model_dir.is_dir():
        raise SystemExit("--model-dir 必须是本地模型目录")
    if args.max_steps < 1 or args.max_length < 8 or args.max_new_tokens < 1:
        raise SystemExit("步数或长度参数无效")
    torch, model_cls, tokenizer_cls, _ = _load_runtime(args.device)
    if args.device.startswith("musa") and not torch.musa.is_available():
        raise RuntimeError("当前运行环境没有可用的 MUSA device")
    tokenizer, model = _load_model(torch, model_cls, tokenizer_cls, args)
    if args.action == "infer":
        _run_infer(torch, tokenizer, model, args)
    else:
        _run_lora(tokenizer, model, args)


if __name__ == "__main__":
    main()
