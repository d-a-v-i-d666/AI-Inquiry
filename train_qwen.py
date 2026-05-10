#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    set_seed,
)

try:
    import jieba  # type: ignore
except ImportError:
    jieba = None

RE_HAN = re.compile(r"([\u4E00-\u9FD5a-zA-Z0-9]+)")

SOURCE_CATEGORIES = [
    "儿科",
    "耳鼻咽喉科",
    "风湿免疫科",
    "妇产科",
    "肝胆外科",
    "感染科 传染科",
    "肛肠外科",
    "骨科",
    "呼吸内科",
    "急诊科",
    "精神心理科",
    "口腔科",
    "泌尿外科",
    "内分泌科",
    "皮肤科",
    "普通内科",
    "普外科",
    "乳腺外科",
    "烧伤科",
    "神经内科",
    "神经外科",
    "疼痛科 麻醉科",
    "头颈外科",
    "消化内科",
    "心血管内科",
    "性病科",
    "胸外科",
    "血液科",
    "眼科",
    "疫苗科",
    "影像检验科",
    "整形科",
    "中医科",
    "肿瘤科",
]

DEFAULT_CONFIDENCE = 0.8


def resolve_path(workspace_dir: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = workspace_dir / path
    return path.resolve()


def normalize_label(label_name: str) -> str:
    return " ".join(label_name.strip().split())


def _segment_content(content: str, use_jieba: bool) -> str:
    content = content.strip()
    if not use_jieba:
        return content

    if jieba is None:
        raise RuntimeError(
            "jieba is required when jieba tokenization is enabled. "
            "Install it with: pip install jieba"
        )

    blocks = RE_HAN.split(content)
    words: List[str] = []
    for blk in blocks:
        if RE_HAN.fullmatch(blk):
            words.extend([w for w in jieba.cut(blk) if w.strip()])

    return " ".join(words) if words else content


def read_category() -> Tuple[List[str], Dict[str, int]]:
    categories = SOURCE_CATEGORIES.copy()
    cat_to_id = dict(zip(categories, range(len(categories))))
    return categories, cat_to_id


def read_csv_split(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "text" not in df.columns:
        raise ValueError(f"Missing 'text' column in {path}")
    if "label_name" not in df.columns:
        raise ValueError(f"Missing 'label_name' column in {path}")

    df = df.dropna(subset=["text", "label_name"]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0]
    df["label_name"] = df["label_name"].astype(str)
    return df


def align_labels(df: pd.DataFrame, label2id: Dict[str, int], split_name: str) -> pd.DataFrame:
    df = df.copy()
    df["label_name"] = df["label_name"].map(normalize_label)

    unknown_labels = sorted(set(df["label_name"].tolist()) - set(label2id.keys()))
    if unknown_labels:
        raise ValueError(
            f"Unknown labels in {split_name} split: {unknown_labels}. "
            "Please update SOURCE_CATEGORIES or clean the data."
        )

    expected_ids = df["label_name"].map(label2id)
    if "label_id" in df.columns:
        df["label_id"] = pd.to_numeric(df["label_id"], errors="coerce")
        mismatched = df["label_id"].notna() & (df["label_id"].astype(int) != expected_ids)
        if mismatched.any():
            mismatch_rows = df.loc[mismatched, ["label_name", "label_id"]].head(5).to_dict("records")
            raise ValueError(
                f"Label id mismatch in {split_name} split. Examples: {mismatch_rows}"
            )

    df["label_id"] = expected_ids.astype(int)
    return df


def build_system_prompt(categories: List[str]) -> str:
    category_list = "、".join(categories)
    return (
        "你是医疗分诊助手。请根据用户描述判断最合适的科室。\n"
        "仅输出严格 JSON，不要输出多余文本或 Markdown。\n"
        "JSON schema: {\"department\": <科室名称>, \"confidence\": <0-1>, \"reason\": <简要理由>}\n"
        f"可选科室列表: {category_list}"
    )


def build_target_json(label_name: str) -> str:
    payload = {
        "department": label_name,
        "confidence": DEFAULT_CONFIDENCE,
        "reason": f"根据症状描述，建议就诊{label_name}。",
    }
    return json.dumps(payload, ensure_ascii=False)


def build_prompt_text(tokenizer, system_prompt: str, user_text: str, add_generation_prompt: bool) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

    if add_generation_prompt:
        return f"System: {system_prompt}\nUser: {user_text}\nAssistant:"
    return f"System: {system_prompt}\nUser: {user_text}\n"


def build_full_text(tokenizer, system_prompt: str, user_text: str, assistant_text: str) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False)

    return f"System: {system_prompt}\nUser: {user_text}\nAssistant: {assistant_text}"


def build_dataset(
    df: pd.DataFrame,
    tokenizer,
    system_prompt: str,
    max_length: int,
    use_jieba: bool,
) -> Dataset:
    if use_jieba:
        df = df.copy()
        df["text"] = df["text"].map(lambda x: _segment_content(x, use_jieba=True))

    ds = Dataset.from_pandas(df[["text", "label_name"]], preserve_index=False)

    def tokenize_batch(batch):
        input_ids = []
        attention_mask = []
        labels = []

        for text, label_name in zip(batch["text"], batch["label_name"]):
            prompt_text = build_prompt_text(
                tokenizer,
                system_prompt,
                user_text=text,
                add_generation_prompt=True,
            )
            target_text = build_target_json(label_name)
            full_text = build_full_text(
                tokenizer,
                system_prompt,
                user_text=text,
                assistant_text=target_text,
            )

            full_tokens = tokenizer(
                full_text,
                truncation=True,
                max_length=max_length,
            )
            prompt_tokens = tokenizer(
                prompt_text,
                truncation=True,
                max_length=max_length,
            )

            full_ids = full_tokens["input_ids"]
            prompt_len = min(len(prompt_tokens["input_ids"]), len(full_ids))
            label_ids = [-100] * prompt_len + full_ids[prompt_len:]

            input_ids.append(full_ids)
            attention_mask.append(full_tokens["attention_mask"])
            labels.append(label_ids)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    return ds.map(tokenize_batch, batched=True, remove_columns=ds.column_names)


def collate_fn(tokenizer, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
    batch = tokenizer.pad(
        {
            "input_ids": [f["input_ids"] for f in features],
            "attention_mask": [f["attention_mask"] for f in features],
        },
        padding=True,
        return_tensors="pt",
    )

    label_tensors = [torch.tensor(f["labels"], dtype=torch.long) for f in features]
    labels = torch.nn.utils.rnn.pad_sequence(label_tensors, batch_first=True, padding_value=-100)
    batch["labels"] = labels
    return batch


def extract_json(text: str) -> Dict[str, object] | None:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    snippet = text[start : end + 1]
    try:
        return json.loads(snippet)
    except json.JSONDecodeError:
        return None


def evaluate_generation(
    model,
    tokenizer,
    df: pd.DataFrame,
    system_prompt: str,
    max_new_tokens: int,
    max_samples: int,
) -> Dict[str, object]:
    categories, _ = read_category()
    label_set = set(categories)

    if max_samples > 0:
        df = df.head(max_samples)

    y_true: List[str] = []
    y_pred: List[str] = []
    failures = []

    model.eval()

    for _, row in df.iterrows():
        user_text = str(row["text"]).strip()
        true_label = normalize_label(str(row["label_name"]))

        prompt_text = build_prompt_text(
            tokenizer,
            system_prompt,
            user_text=user_text,
            add_generation_prompt=True,
        )

        encoded = tokenizer(prompt_text, return_tensors="pt")
        encoded = {k: v.to(model.device) for k, v in encoded.items()}

        with torch.no_grad():
            output_ids = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
            )

        generated_ids = output_ids[0][encoded["input_ids"].shape[1] :]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        payload = extract_json(generated_text)

        pred_label = None
        if isinstance(payload, dict):
            pred_label = payload.get("department") or payload.get("科室")

        y_true.append(true_label)
        y_pred.append(pred_label if isinstance(pred_label, str) else "__invalid__")

        if pred_label not in label_set:
            failures.append(
                {
                    "text": user_text,
                    "true_label": true_label,
                    "pred_label": pred_label,
                    "raw_output": generated_text,
                }
            )

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=categories,
        average="macro",
        zero_division=0,
    )

    return {
        "samples": len(y_true),
        "accuracy": acc,
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
        "parse_failures": failures[:50],
        "parse_failure_count": len(failures),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning for Qwen3.6 triage")
    parser.add_argument("--workspace-dir", default=".")
    parser.add_argument("--model-id", default="Qwen/Qwen3.6-27B")
    parser.add_argument("--cache-dir", default=".hf-cache")
    parser.add_argument("--train-csv", default="data_csv/train.csv")
    parser.add_argument("--val-csv", default="data_csv/val.csv")
    parser.add_argument("--test-csv", default="data_csv/test.csv")
    parser.add_argument("--output-dir", default="outputs/qwen3.6-27b-qlora")
    parser.add_argument("--adapter-dir", default="models/qwen3.6-27b-qlora")

    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--num-train-epochs", type=int, default=3)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disable-jieba", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--run-eval", action="store_true")
    parser.add_argument("--eval-split", default="val", choices=["val", "test"])
    parser.add_argument("--max-eval-samples", type=int, default=200)

    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )

    args = parser.parse_args()

    workspace_dir = Path(args.workspace_dir).expanduser().resolve()
    cache_dir = resolve_path(workspace_dir, args.cache_dir)
    output_dir = resolve_path(workspace_dir, args.output_dir)
    adapter_dir = resolve_path(workspace_dir, args.adapter_dir)
    train_path = resolve_path(workspace_dir, args.train_csv)
    val_path = resolve_path(workspace_dir, args.val_csv)
    test_path = resolve_path(workspace_dir, args.test_csv)

    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir.mkdir(parents=True, exist_ok=True)

    use_jieba = not args.disable_jieba

    set_seed(args.seed)

    _, label2id = read_category()
    train_df = align_labels(read_csv_split(train_path), label2id, split_name="train")
    val_df = align_labels(read_csv_split(val_path), label2id, split_name="val")
    test_df = align_labels(read_csv_split(test_path), label2id, split_name="test")

    system_prompt = build_system_prompt(list(label2id.keys()))

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        cache_dir=str(cache_dir),
        trust_remote_code=args.trust_remote_code,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        cache_dir=str(cache_dir),
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=args.trust_remote_code,
    )

    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    lora_targets = [name.strip() for name in args.lora_target_modules.split(",") if name.strip()]
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=lora_targets,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)

    train_ds = build_dataset(
        train_df,
        tokenizer=tokenizer,
        system_prompt=system_prompt,
        max_length=args.max_length,
        use_jieba=use_jieba,
    )
    val_ds = build_dataset(
        val_df,
        tokenizer=tokenizer,
        system_prompt=system_prompt,
        max_length=args.max_length,
        use_jieba=use_jieba,
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_strategy="epoch",
        save_total_limit=args.save_total_limit,
        evaluation_strategy="epoch",
        report_to="none",
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        remove_unused_columns=False,
        optim="paged_adamw_8bit",
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=lambda features: collate_fn(tokenizer, features),
    )

    train_result = trainer.train()
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    run_summary = {
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
        "max_length": args.max_length,
        "gradient_accumulation": args.gradient_accumulation,
        "train_metrics": train_result.metrics,
        "adapter_dir": str(adapter_dir),
        "system_prompt": system_prompt,
        "use_jieba": use_jieba,
    }

    summary_path = output_dir / "run_summary.json"
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(run_summary, file, ensure_ascii=False, indent=2)

    print("Training finished.")
    print(f"Adapter saved to: {adapter_dir}")
    print(f"Run summary saved to: {summary_path}")

    if args.run_eval:
        eval_df = val_df if args.eval_split == "val" else test_df
        eval_result = evaluate_generation(
            model=model,
            tokenizer=tokenizer,
            df=eval_df,
            system_prompt=system_prompt,
            max_new_tokens=args.max_new_tokens,
            max_samples=args.max_eval_samples,
        )
        eval_path = output_dir / f"eval_{args.eval_split}.json"
        with open(eval_path, "w", encoding="utf-8") as file:
            json.dump(eval_result, file, ensure_ascii=False, indent=2)
        print(f"Eval results saved to: {eval_path}")


if __name__ == "__main__":
    main()
