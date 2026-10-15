"""MiniCPM text inference and LoRA entry point for MUSA."""

import argparse
import json
import os
from pathlib import Path


def _prompt(text):
    return f"<用户>{text}<AI>"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["infer", "lora"])
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--prompt", default="请介绍一下你自己。")
    parser.add_argument("--data-json", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--dtype", choices=["float32", "bf16"], default="bf16")
    parser.add_argument("--device", default="musa:0")
    args = parser.parse_args()
    if not args.model_dir.is_dir():
        parser.error("--model-dir 必须是本地模型目录")
    if args.adapter_path and not args.adapter_path.is_dir():
        parser.error("--adapter-path 必须是本地目录")
    if args.action == "lora" and (args.data_json is None or args.output_dir is None):
        parser.error("lora 需要 --data-json 和 --output-dir")
    if not args.device.startswith("musa"):
        parser.error("此入口要求 --device 使用 musa[:index]")

    os.environ.setdefault("TORCHADA_FORCE_CUDA_AVAILABLE", "1")
    import torch_musa  # noqa: F401
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.musa.is_available():
        raise RuntimeError("请在 MUSA 容器中运行")
    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.adapter_path or args.model_dir), local_files_only=True,
        use_fast=False, trust_remote_code=True,
    )
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model_dir), local_files_only=True, trust_remote_code=True,
        torch_dtype=dtype,
    ).to(args.device)
    if args.adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(args.adapter_path))
    model.eval()
    encoded = tokenizer(_prompt(args.prompt), return_tensors="pt")
    encoded = {key: value.to(args.device) for key, value in encoded.items()}
    with torch.inference_mode():
        generated = model.generate(
            **encoded, max_new_tokens=args.max_new_tokens, do_sample=False,
            eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
        )
    answer = tokenizer.decode(
        generated[0, encoded["input_ids"].shape[1]:], skip_special_tokens=True
    )
    print(answer)
    print(json.dumps({"action": "infer", "status": "PASS", "device": args.device}))


if __name__ == "__main__":
    main()
