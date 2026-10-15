"""MiniCPM-o 2.6 text/image inference entry point for MUSA."""

import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--prompt", default="请介绍一下你自己。")
    parser.add_argument("--image", type=Path)
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--dtype", choices=["float32", "bf16"], default="bf16")
    parser.add_argument("--device", default="musa:0")
    args = parser.parse_args()
    if not args.model_dir.is_dir():
        parser.error("--model-dir 必须是本地模型目录")
    if args.image and not args.image.is_file():
        parser.error("--image 必须是本地图片")
    if args.adapter_path and not args.adapter_path.is_dir():
        parser.error("--adapter-path 必须是本地 adapter 目录")
    if not args.device.startswith("musa"):
        parser.error("此入口要求 --device 使用 musa[:index]")

    os.environ.setdefault("TORCHADA_FORCE_CUDA_AVAILABLE", "1")
    import torch_musa  # noqa: F401
    import torch
    from PIL import Image
    from transformers import AutoModel, AutoTokenizer

    if not torch.musa.is_available():
        raise RuntimeError("请在 MUSA 容器中运行")
    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    model = AutoModel.from_pretrained(
        str(args.model_dir), local_files_only=True, trust_remote_code=True,
        torch_dtype=dtype, attn_implementation="sdpa",
    ).eval().to(args.device)
    if args.adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(args.adapter_path))
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model_dir), local_files_only=True, trust_remote_code=True,
    )
    image = Image.open(args.image).convert("RGB") if args.image else None
    response = model.chat(
        image=image,
        msgs=[{"role": "user", "content": args.prompt}],
        tokenizer=tokenizer,
        sampling=False,
        max_new_tokens=args.max_new_tokens,
    )
    print(response)
    print(json.dumps({"action": "infer", "status": "PASS", "device": args.device}))


if __name__ == "__main__":
    main()
