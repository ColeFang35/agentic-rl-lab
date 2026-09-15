# -*- coding: utf-8 -*-
"""L1 · RFT：在"成功轨迹"上做 QLoRA 指令微调（拒绝采样的核心一步）。

不写任何 RL 算法——只用「筛出成功的、再学一遍」把成功率顶上去。
这是 agentic RL 最实用、最容易落地的形态。
"""
from __future__ import annotations

import argparse

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          DataCollatorForSeq2Seq, Trainer, TrainingArguments)


def args_() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="RFT：在成功轨迹上做指令微调")
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="data/sft.jsonl")
    ap.add_argument("--output-dir", default="outputs/rft-adapter")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-len", type=int, default=768)
    ap.add_argument("--lora-r", type=int, default=8)
    return ap.parse_args()


def main() -> None:
    a = args_()
    torch.manual_seed(42)
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    raw = load_dataset("json", data_files=a.data, split="train")
    # prompt 已经是渲染好的 chat 文本（含 system + few-shot），直接拼接
    def tok_fn(ex):
        p = tok(ex["prompt"], add_special_tokens=False)["input_ids"]
        c = tok(ex["completion"] + tok.eos_token, add_special_tokens=False)["input_ids"]
        ids = (p + c)[: a.max_len]
        labels = ([-100] * len(p) + c)[: a.max_len]      # 只对 completion 算 loss
        return {"input_ids": ids, "labels": labels, "attention_mask": [1] * len(ids)}

    ds = raw.map(tok_fn, remove_columns=raw.column_names)
    print(f"RFT 训练样本：{len(ds)} 条")

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(a.model, quantization_config=bnb,
                                                 device_map="auto", trust_remote_code=True)
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(
        r=a.lora_r, lora_alpha=a.lora_r * 2, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()

    common = dict(output_dir=a.output_dir + "/ckpt", num_train_epochs=a.epochs,
                  per_device_train_batch_size=a.batch_size,
                  gradient_accumulation_steps=a.grad_accum, learning_rate=a.lr,
                  lr_scheduler_type="cosine", logging_steps=10, save_strategy="epoch",
                  save_total_limit=1, bf16=torch.cuda.is_bf16_supported(),
                  report_to=[], seed=42)
    try:
        targs = TrainingArguments(warmup_ratio=0.03, **common)
    except TypeError:
        targs = TrainingArguments(warmup_steps=10, **common)

    Trainer(model=model, args=targs, train_dataset=ds,
            data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100)
            ).train()
    model.save_pretrained(a.output_dir); tok.save_pretrained(a.output_dir)
    print(f"RFT 适配器已保存：{a.output_dir}")


if __name__ == "__main__":
    main()
