# -*- coding: utf-8 -*-
"""L2 · DPO：用「同题最优 vs 最差」的自动偏好对做偏好优化。

为什么要 DPO 而不是 PPO：
- PPO 需要额外训一个 value 网络 + 在线采样，工程与算力成本高；
- DPO 直接在偏好对上做**离线**优化，等价于隐式奖励建模，小规模实验性价比高得多。
"""
from __future__ import annotations

import argparse
import json

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

try:                                        # trl 新版
    from trl import DPOConfig as _DPOCfg, DPOTrainer as _DPOTrainer
    _NEW = True
except ImportError:                          # trl 老版
    from transformers import TrainingArguments as _DPOCfg
    from trl import DPOTrainer as _DPOTrainer
    _NEW = False


def args_() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="DPO 偏好优化")
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="data/dpo_pairs.jsonl")
    ap.add_argument("--output-dir", default="outputs/dpo-adapter")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--beta", type=float, default=0.1, help="KL 约束强度，越大越保守")
    ap.add_argument("--max-len", type=int, default=768)
    ap.add_argument("--lora-r", type=int, default=8)
    return ap.parse_args()


def main() -> None:
    a = args_()
    torch.manual_seed(42)
    rows = [json.loads(l) for l in open(a.data, encoding="utf-8")]
    # trl 只认这三列，去掉我们自己的元信息
    ds = Dataset.from_list([{"prompt": r["prompt"], "chosen": r["chosen"],
                             "rejected": r["rejected"]} for r in rows])
    print(f"DPO 偏好对：{len(ds)} 条")

    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(a.model, quantization_config=bnb,
                                                 device_map="auto", trust_remote_code=True)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(
        r=a.lora_r, lora_alpha=a.lora_r * 2, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()

    common = dict(output_dir=a.output_dir + "/ckpt", num_train_epochs=a.epochs,
                  per_device_train_batch_size=a.batch_size,
                  gradient_accumulation_steps=a.grad_accum, learning_rate=a.lr,
                  lr_scheduler_type="cosine", logging_steps=5, save_strategy="epoch",
                  save_total_limit=1, bf16=torch.cuda.is_bf16_supported(),
                  report_to=[], seed=42)
    if _NEW:
        common.update(beta=a.beta, max_length=a.max_len)
        targs = _DPOCfg(**common)
        trainer = _DPOTrainer(model=model, args=targs, train_dataset=ds, processing_class=tok)
    else:
        targs = _DPOCfg(warmup_ratio=0.03, **common)
        trainer = _DPOTrainer(model=model, args=targs, train_dataset=ds,
                              tokenizer=tok, beta=a.beta, max_length=a.max_len)
    trainer.train()
    model.save_pretrained(a.output_dir); tok.save_pretrained(a.output_dir)
    print(f"DPO 适配器已保存：{a.output_dir}")


if __name__ == "__main__":
    main()
