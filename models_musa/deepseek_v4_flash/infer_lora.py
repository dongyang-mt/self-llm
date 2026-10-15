"""DeepSeek-V4-Flash LoRA inference entry point for MUSA."""

import argparse
import json
import os
from pathlib import Path


def _prompt(system_prompt, user_prompt):
    return (
        "<｜begin▁of▁sentence｜>"
        f"{system_prompt}<｜User｜>{user_prompt}<｜Assistant｜></think>"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--prompt", default="你是谁？")
    parser.add_argument("--system-prompt", default="现在你要扮演皇帝身边的女人--甄嬛。")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--dtype", choices=["float32", "bf16"], default="bf16")
    parser.add_argument("--device", default="musa:0")
    args = parser.parse_args()
    if not args.model_path.is_dir():
        parser.error("--model-path 必须是本地模型目录")
    if args.adapter_path is not None and not args.adapter_path.is_dir():
        parser.error("--adapter-path 必须是本地 adapter 目录")
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens 必须大于 0")
    if not args.device.startswith("musa"):
        parser.error("此入口要求 --device 使用 musa[:index]")

    os.environ.setdefault("TORCHADA_FORCE_CUDA_AVAILABLE", "1")
    import torch_musa  # noqa: F401
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.musa.is_available():
        raise RuntimeError("请在 MUSA 容器中运行")
    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    tokenizer_path = args.adapter_path or args.model_path
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path), local_files_only=True, trust_remote_code=True
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model_path),
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=dtype,
    ).to(args.device)
    if args.adapter_path is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(args.adapter_path))
    model.eval()
    encoded = tokenizer(
        _prompt(args.system_prompt, args.prompt), return_tensors="pt"
    )
    encoded = {key: value.to(args.device) for key, value in encoded.items()}
    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    answer = tokenizer.decode(
        generated[0, encoded["input_ids"].shape[1] :], skip_special_tokens=True
    )
    print(answer.split("<｜end▁of▁sentence｜>", 1)[0].strip())
    print(json.dumps({"action": "infer", "status": "PASS", "device": args.device}))


if __name__ == "__main__":
    main()
