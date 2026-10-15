"""Run a Gemma 4 E4B LoRA adapter on a MUSA device.

The CUDA tutorial under ``models/Gemma4`` loads the model with
``device_map='auto'``.  Accelerate's automatic device mapping does not
reliably identify the MUSA backend, so this entry point loads the model on
the host and moves it explicitly to the requested device.
"""

from __future__ import annotations

import argparse
import os
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, help="Base Gemma 4 model directory")
    parser.add_argument("--lora-path", required=True, help="PEFT LoRA adapter directory")
    parser.add_argument("--prompt", default="你是谁？", help="User prompt")
    parser.add_argument(
        "--system-prompt",
        default="现在你要扮演皇帝身边的女人--甄嬛",
        help="System prompt used by the Gemma chat template",
    )
    parser.add_argument("--device", default="musa", help="Torch device, for example musa:0 or cpu")
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
        help="Model parameter dtype",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=1)
    return parser.parse_args()


def _load_runtime(device_name: str) -> tuple[Any, Any, Any, Any]:
    """Import the runtime after selecting the backend.

    ``TORCHADA_FORCE_CUDA_AVAILABLE`` must be set before importing torch and
    the Transformers stack when torchada is used as the MUSA compatibility
    layer.  Importing ``torch_musa`` registers the ``musa`` device type.
    """

    if device_name.startswith("musa"):
        os.environ.setdefault("TORCHADA_FORCE_CUDA_AVAILABLE", "1")
        try:
            import torch_musa  # noqa: F401  # registers the MUSA backend
        except ImportError as exc:
            raise RuntimeError(
                "MUSA inference requires torch_musa in the runtime environment"
            ) from exc

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    return torch, PeftModel, AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    args = _parse_args()
    torch, PeftModel, AutoModelForCausalLM, AutoTokenizer = _load_runtime(args.device)

    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]
    device = torch.device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, model_id=args.lora_path)
    model = model.to(device).eval()

    inputs = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": args.system_prompt},
            {"role": "user", "content": args.prompt},
        ],
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
    ).to(device)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            top_k=args.top_k,
        )

    prompt_length = inputs["input_ids"].shape[1]
    answer = tokenizer.decode(output_ids[0, prompt_length:], skip_special_tokens=True)
    print(f"prompt: {args.prompt}")
    print(f"system_prompt: {args.system_prompt}")
    print(f"output: {answer}")


if __name__ == "__main__":
    main()
