"""LLaMA3-8B-Instruct Transformers inference and LoRA entry for MUSA.

The original notebook remains unchanged.  This file keeps the same local
model/data paths while adding the MUSA device setup in one isolated adapter.
"""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TORCHADA_FORCE_CUDA_AVAILABLE", "1")
import torchada  # noqa: F401
import torch_musa  # noqa: F401


def _chat_text(tokenizer, prompt):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


def _load_model(model_dir, dtype):
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        local_files_only=True,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to("cuda:0")
    if any(p.device.type != "musa" or p.device.index != 0 for p in model.parameters()):
        raise RuntimeError("模型参数未全部位于 MUSA device 0")
    return model


def run_infer(args, tokenizer, model):
    import torch

    inputs = tokenizer(_chat_text(tokenizer, args.prompt), return_tensors="pt").to("cuda:0")
    model.eval()
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
        )
    if output.device.type != "musa":
        raise RuntimeError("生成结果未位于 MUSA")
    answer = tokenizer.decode(output[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
    print(answer)
    print(json.dumps({"action": "infer", "status": "PASS", "device": str(output.device)}))


def run_lora(args, tokenizer, model):
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments

    records = json.loads(args.data_json.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise ValueError("data-json 必须是非空数组")
    examples = []
    for record in records:
        prompt_ids = tokenizer(
            _chat_text(tokenizer, record["instruction"] + record.get("input", "")),
            add_special_tokens=False,
        )["input_ids"]
        response_ids = tokenizer(record["output"], add_special_tokens=False)["input_ids"]
        response_ids.append(tokenizer.eos_token_id)
        ids = (prompt_ids + response_ids)[: args.max_length]
        labels = ([-100] * len(prompt_ids) + response_ids)[: args.max_length]
        if not any(label != -100 for label in labels):
            raise ValueError("max-length 太小，未保留答案 token")
        examples.append({"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels})

    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
    ))
    model.enable_input_require_grads()
    model.config.use_cache = False
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=1,
        max_steps=args.max_steps,
        learning_rate=1e-4,
        logging_steps=1,
        save_strategy="no",
        bf16=args.dtype == "bf16",
        fp16=False,
        gradient_checkpointing=True,
        report_to="none",
        dataloader_num_workers=0,
    )

    class MusaTrainer(Trainer):
        def _get_train_sampler(self, train_dataset=None):
            from torch.utils.data import RandomSampler
            import torch

            dataset = self.train_dataset if train_dataset is None else train_dataset
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self.args.seed)
            return RandomSampler(dataset, generator=generator)

    trainer = MusaTrainer(
        model=model,
        args=training_args,
        train_dataset=examples,
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True),
    )
    result = trainer.train()
    model.save_pretrained(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    print(json.dumps({
        "action": "lora",
        "status": "PASS",
        "device": str(next(model.parameters()).device),
        "steps": result.global_step,
        "loss": result.training_loss,
        "adapter_directory": str(args.output_dir),
    }, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["infer", "lora"])
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--prompt", default="请介绍一下你自己。")
    parser.add_argument("--data-json", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--dtype", choices=["float32", "bf16"], default="bf16")
    args = parser.parse_args()
    if not args.model_dir.is_dir():
        parser.error("--model-dir 必须是本地模型目录")
    if args.action == "lora" and (args.data_json is None or args.output_dir is None):
        parser.error("lora 需要 --data-json 和 --output-dir")
    if args.max_steps < 1 or args.max_length < 8 or args.max_new_tokens < 1:
        parser.error("步数或长度参数无效")

    import torch
    from transformers import AutoTokenizer

    if not torch.musa.is_available() or not torch.cuda.is_available():
        raise RuntimeError("请在 MUSA 容器中运行")
    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model_dir), local_files_only=True, use_fast=False, trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = _load_model(args.model_dir, dtype)
    if args.action == "infer":
        run_infer(args, tokenizer, model)
    else:
        run_lora(args, tokenizer, model)


if __name__ == "__main__":
    main()
