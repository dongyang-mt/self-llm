"""Run Qwen3-VL-4B-Instruct (optionally with a PEFT adapter) on MUSA."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--lora-dir", type=Path)
    parser.add_argument("--prompt", default="请介绍一下你自己。")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    args = parser.parse_args()
    if not args.model_dir.is_dir():
        parser.error("--model-dir must be a local model directory")
    if args.lora_dir is not None and not args.lora_dir.is_dir():
        parser.error("--lora-dir must be a local PEFT adapter directory")

    os.environ.setdefault("TORCHADA_FORCE_CUDA_AVAILABLE", "1")
    import torchada  # noqa: F401  # registers CUDA-compatible MUSA hooks
    import torch_musa  # noqa: F401  # registers the MUSA device
    import torch
    from transformers import AutoProcessor

    try:
        from transformers import Qwen3VLForConditionalGeneration
    except ImportError:
        from transformers import AutoModelForVision2Seq as Qwen3VLForConditionalGeneration

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(args.model_dir), torch_dtype=dtype, device_map=None, local_files_only=True
    ).to("cuda:0").eval()
    if args.lora_dir is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(args.lora_dir)).to("cuda:0").eval()

    if any(parameter.device.type != "musa" or parameter.device.index != 0 for parameter in model.parameters()):
        raise RuntimeError("Model parameters did not land on MUSA device 0")
    processor = AutoProcessor.from_pretrained(str(args.model_dir), local_files_only=True)
    messages = [{"role": "user", "content": [{"type": "text", "text": args.prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], return_tensors="pt").to("cuda:0")
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
    if output.device.type != "musa":
        raise RuntimeError("Generation output did not land on MUSA")
    prompt_length = inputs["input_ids"].shape[1]
    answer = processor.batch_decode(output[:, prompt_length:], skip_special_tokens=True)[0]
    print(answer)
    print({"device": str(output.device), "status": "PASS"})


if __name__ == "__main__":
    main()
