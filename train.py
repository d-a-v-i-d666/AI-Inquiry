#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from huggingface_hub import snapshot_download
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
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


def resolve_path(workspace_dir: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = workspace_dir / path
    return path.resolve()


def has_local_model(model_dir: Path) -> bool:
    has_config = (model_dir / "config.json").exists()
    has_vocab = (model_dir / "vocab.txt").exists() or (model_dir / "tokenizer.json").exists()
    has_weights = (model_dir / "pytorch_model.bin").exists() or (model_dir / "model.safetensors").exists()
    return has_config and has_vocab and has_weights


def ensure_local_base_model(model_id: str, model_dir: Path, cache_dir: Path) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if has_local_model(model_dir):
        return

    snapshot_download(
        repo_id=model_id,
        local_dir=str(model_dir),
        cache_dir=str(cache_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
    )


def read_category() -> Tuple[List[str], Dict[str, int]]:
    categories = SOURCE_CATEGORIES.copy()
    cat_to_id = dict(zip(categories, range(len(categories))))
    return categories, cat_to_id


def normalize_label(label_name: str) -> str:
    return " ".join(label_name.strip().split())


def _parse_source_line(line: str) -> Tuple[str, str] | None:
    line = line.rstrip("\n").rstrip("\r")
    if not line:
        return None

    parts = [part.strip() for part in line.split("\t")]
    non_empty_parts = [part for part in parts if part]
    if len(non_empty_parts) != 2:
        return None

    label_name, content = non_empty_parts
    if not label_name or not content:
        return None

    return normalize_label(label_name), content


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


def read_file(
    filename: Path,
    use_jieba: bool,
    encodings=("utf-8-sig", "utf-8", "gb18030", "gbk"),
) -> Tuple[List[str], List[str]]:
    last_error = None
    for enc in encodings:
        try:
            labels: List[str] = []
            contents: List[str] = []
            with open(filename, "r", encoding=enc) as file:
                for line in file:
                    parsed = _parse_source_line(line)
                    if parsed is None:
                        continue

                    label_name, content = parsed
                    labels.append(label_name)
                    contents.append(_segment_content(content, use_jieba=use_jieba))

            if labels:
                return labels, contents
        except UnicodeDecodeError as exc:
            last_error = exc

    if last_error is not None:
        raise last_error

    raise RuntimeError(f"No valid rows found in {filename}")


def read_text_split(path: Path, use_jieba: bool) -> pd.DataFrame:
    labels, contents = read_file(path, use_jieba=use_jieba)
    return pd.DataFrame({"label_name": labels, "text": contents})


def add_numeric_labels(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, int], Dict[int, str]]:
    categories, label2id = read_category()
    id2label = {idx: name for name, idx in label2id.items()}

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    train_df["label_name"] = train_df["label_name"].map(normalize_label)
    val_df["label_name"] = val_df["label_name"].map(normalize_label)
    test_df["label_name"] = test_df["label_name"].map(normalize_label)

    for split_name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        unknown_labels = sorted(set(df["label_name"].tolist()) - set(categories))
        if unknown_labels:
            raise ValueError(
                f"Unknown labels in {split_name} split: {unknown_labels}. "
                "Please update SOURCE_CATEGORIES or clean the data."
            )

    train_df["label"] = train_df["label_name"].map(label2id).astype(int)
    val_df["label"] = val_df["label_name"].map(label2id).astype(int)
    test_df["label"] = test_df["label_name"].map(label2id).astype(int)

    return train_df, val_df, test_df, label2id, id2label


def dataframe_to_dataset(df: pd.DataFrame) -> Dataset:
    return Dataset.from_pandas(df[["text", "label"]], preserve_index=False)


def build_training_arguments(args: argparse.Namespace, output_dir: Path) -> TrainingArguments:
    common_kwargs = dict(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        save_strategy="epoch",
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=args.logging_steps,
        report_to="none",
        fp16=torch.cuda.is_available() and (not args.disable_fp16),
        seed=args.seed,
    )

    try:
        return TrainingArguments(eval_strategy="epoch", **common_kwargs)
    except TypeError:
        return TrainingArguments(evaluation_strategy="epoch", **common_kwargs)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    return {
        "accuracy": acc,
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
    }


def save_class_weights(train_df: pd.DataFrame, id2label: Dict[int, str], output_dir: Path) -> Path:
    num_labels = len(id2label)
    total = len(train_df)
    label_counts = train_df["label"].value_counts().reindex(range(num_labels), fill_value=0)

    rows = []
    for label_idx, count in label_counts.items():
        if count > 0:
            class_weight = total / (num_labels * count)
        else:
            class_weight = 0.0
        rows.append(
            {
                "label_idx": int(label_idx),
                "department": id2label[int(label_idx)],
                "train_count": int(count),
                "class_weight": float(class_weight),
            }
        )

    class_weights_df = pd.DataFrame(rows).sort_values("class_weight", ascending=False)
    class_weights_path = output_dir / "class_weights.csv"
    class_weights_df.to_csv(class_weights_path, index=False, encoding="utf-8-sig")
    return class_weights_path


def save_loss_curve(log_df: pd.DataFrame, output_dir: Path) -> Path | None:
    train_loss_df = pd.DataFrame()
    eval_loss_df = pd.DataFrame()

    if "loss" in log_df.columns:
        train_loss_df = log_df[log_df["loss"].notna()].copy()
    if "eval_loss" in log_df.columns:
        eval_loss_df = log_df[log_df["eval_loss"].notna()].copy()

    if train_loss_df.empty and eval_loss_df.empty:
        return None

    plt.figure(figsize=(8, 5))

    if not train_loss_df.empty:
        x_train = (
            train_loss_df["epoch"]
            if "epoch" in train_loss_df.columns
            else np.arange(1, len(train_loss_df) + 1)
        )
        plt.plot(x_train, train_loss_df["loss"], marker="o", linewidth=1.5, label="train_loss")

    if not eval_loss_df.empty:
        x_eval = (
            eval_loss_df["epoch"]
            if "epoch" in eval_loss_df.columns
            else np.arange(1, len(eval_loss_df) + 1)
        )
        plt.plot(x_eval, eval_loss_df["eval_loss"], marker="s", linewidth=1.5, label="eval_loss")

    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Training / Validation Loss Curve")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    loss_curve_path = output_dir / "loss_curve.png"
    plt.savefig(loss_curve_path, dpi=200)
    plt.close()
    return loss_curve_path


def run_single_prediction(
    finetuned_dir: Path,
    tokenizer,
    id2label: Dict[int, str],
    max_length: int,
    sample_text: str,
) -> Dict[str, object]:
    inference_model = AutoModelForSequenceClassification.from_pretrained(
        str(finetuned_dir),
        local_files_only=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inference_model.to(device)
    inference_model.eval()

    encoded = tokenizer(
        sample_text,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        logits = inference_model(**encoded).logits
        probs = torch.softmax(logits, dim=-1)
        pred_idx = int(torch.argmax(probs, dim=-1).item())
        confidence = float(probs[0, pred_idx].item())

    return {
        "pred_idx": pred_idx,
        "pred_department": id2label[pred_idx],
        "confidence": round(confidence, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune mc-bert with files from data/ directory")
    parser.add_argument("--workspace-dir", default=".")
    parser.add_argument("--model-id", default="freedomking/mc-bert")
    parser.add_argument("--local-model-dir", default="models/mc-bert")
    parser.add_argument("--cache-dir", default=".hf-cache")
    parser.add_argument("--train-file", default="data/v2.0Train.txt")
    parser.add_argument("--val-file", default="data/v2.0Val.txt")
    parser.add_argument("--test-file", default="data/v2.0Test.txt")
    parser.add_argument("--output-dir", default="outputs/mc-bert-data")
    parser.add_argument("--finetuned-dir", default="models/mc-bert-data")

    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--train-batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--num-train-epochs", type=float, default=200)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disable-fp16", action="store_true")
    parser.add_argument("--disable-jieba", action="store_true")
    parser.add_argument("--sample-text", default="最近总是腹泻，吃什么药好？")

    args = parser.parse_args()

    workspace_dir = Path(args.workspace_dir).expanduser().resolve()
    local_model_dir = resolve_path(workspace_dir, args.local_model_dir)
    cache_dir = resolve_path(workspace_dir, args.cache_dir)
    train_path = resolve_path(workspace_dir, args.train_file)
    val_path = resolve_path(workspace_dir, args.val_file)
    test_path = resolve_path(workspace_dir, args.test_file)
    output_dir = resolve_path(workspace_dir, args.output_dir)
    finetuned_dir = resolve_path(workspace_dir, args.finetuned_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    finetuned_dir.mkdir(parents=True, exist_ok=True)

    use_jieba = not args.disable_jieba

    set_seed(args.seed)

    ensure_local_base_model(args.model_id, local_model_dir, cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(str(local_model_dir), local_files_only=True)

    train_df = read_text_split(train_path, use_jieba=use_jieba)
    val_df = read_text_split(val_path, use_jieba=use_jieba)
    test_df = read_text_split(test_path, use_jieba=use_jieba)

    train_df, val_df, test_df, label2id, id2label = add_numeric_labels(train_df, val_df, test_df)
    num_labels = len(label2id)

    print(f"Train size: {len(train_df)} | Val size: {len(val_df)} | Test size: {len(test_df)}")
    print(f"Num labels: {num_labels} | jieba tokenization: {use_jieba}")

    label_map_path = output_dir / "label_mapping.json"
    with open(label_map_path, "w", encoding="utf-8") as file:
        json.dump(
            {
                "label2id": label2id,
                "id2label": {str(k): v for k, v in id2label.items()},
                "use_jieba": use_jieba,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    class_weights_path = save_class_weights(train_df, id2label, output_dir)

    train_ds = dataframe_to_dataset(train_df)
    val_ds = dataframe_to_dataset(val_df)
    test_ds = dataframe_to_dataset(test_df)

    def tokenize_batch(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.max_length,
        )

    train_ds = train_ds.map(tokenize_batch, batched=True, remove_columns=["text"])
    val_ds = val_ds.map(tokenize_batch, batched=True, remove_columns=["text"])
    test_ds = test_ds.map(tokenize_batch, batched=True, remove_columns=["text"])

    classifier_model = AutoModelForSequenceClassification.from_pretrained(
        str(local_model_dir),
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        local_files_only=True,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    training_args = build_training_arguments(args, output_dir)

    trainer = Trainer(
        model=classifier_model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
    )

    train_result = trainer.train()
    val_metrics = trainer.evaluate(eval_dataset=val_ds, metric_key_prefix="val")
    test_metrics = trainer.evaluate(eval_dataset=test_ds, metric_key_prefix="test")

    trainer.save_model(str(finetuned_dir))
    tokenizer.save_pretrained(str(finetuned_dir))

    log_history = trainer.state.log_history
    log_df = pd.DataFrame(log_history)
    log_csv_path = output_dir / "trainer_log_history.csv"
    log_df.to_csv(log_csv_path, index=False, encoding="utf-8-sig")

    loss_curve_path = save_loss_curve(log_df, output_dir)

    run_summary = {
        "workspace_dir": str(workspace_dir),
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
        "num_labels": num_labels,
        "use_jieba": use_jieba,
        "best_global_step": trainer.state.best_global_step,
        "best_metric": trainer.state.best_metric,
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
        "train_metrics": train_result.metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "artifacts": {
            "label_mapping": str(label_map_path),
            "class_weights": str(class_weights_path),
            "trainer_log_history": str(log_csv_path),
            "loss_curve": str(loss_curve_path) if loss_curve_path is not None else None,
            "finetuned_model_dir": str(finetuned_dir),
        },
    }

    summary_path = output_dir / "run_summary.json"
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(run_summary, file, ensure_ascii=False, indent=2)

    print("Training finished.")
    print(f"Class weights saved to: {class_weights_path}")
    print(f"Log history saved to: {log_csv_path}")
    if loss_curve_path is not None:
        print(f"Loss curve saved to: {loss_curve_path}")
    print(f"Fine-tuned model saved to: {finetuned_dir}")
    print(f"Run summary saved to: {summary_path}")

    if args.sample_text:
        sample_text = _segment_content(args.sample_text, use_jieba=use_jieba)
        pred = run_single_prediction(
            finetuned_dir=finetuned_dir,
            tokenizer=tokenizer,
            id2label=id2label,
            max_length=args.max_length,
            sample_text=sample_text,
        )
        print("Sample prediction:", pred)


if __name__ == "__main__":
    main()
