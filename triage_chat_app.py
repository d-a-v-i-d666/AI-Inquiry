#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

try:
    import jieba  # type: ignore
except ImportError:
    jieba = None

RE_HAN = re.compile(r"([\u4E00-\u9FD5a-zA-Z0-9]+)")


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def segment_text(text: str, use_jieba: bool) -> str:
    text = text.strip()
    if not text:
        return ""

    if not use_jieba:
        return text

    if jieba is None:
        raise RuntimeError("jieba is required. Install it with: pip install jieba")

    blocks = RE_HAN.split(text)
    words: List[str] = []
    for blk in blocks:
        if RE_HAN.fullmatch(blk):
            words.extend([w for w in jieba.cut(blk) if w.strip()])

    return " ".join(words) if words else text


def resolve_best_model_dir(experiment_dir: Path) -> Tuple[Path, Dict[str, Any]]:
    summary_path = experiment_dir / "run_summary.json"
    if summary_path.exists():
        summary = load_json(summary_path)
        best_model = summary.get("best_model_checkpoint")
        if isinstance(best_model, str) and best_model:
            best_model_dir = Path(best_model)
            if best_model_dir.exists():
                return best_model_dir, summary

    checkpoint_dirs = sorted(
        [p for p in experiment_dir.glob("checkpoint-*") if p.is_dir()],
        key=lambda x: int(x.name.split("-")[-1]),
    )
    if checkpoint_dirs:
        return checkpoint_dirs[-1], {}

    raise FileNotFoundError(
        f"No available checkpoint found under experiment directory: {experiment_dir}"
    )


class DepartmentClassifier:
    def __init__(
        self,
        model_dir: Path,
        use_jieba: bool,
        max_length: int,
        top_k: int,
    ):
        self.model_dir = model_dir
        self.use_jieba = use_jieba
        self.max_length = max_length
        self.top_k = top_k

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(model_dir),
            local_files_only=True,
        )
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text: str, top_k: Optional[int] = None) -> Dict[str, Any]:
        normalized = segment_text(text, use_jieba=self.use_jieba)
        if not normalized:
            raise ValueError("Input text is empty")

        encoded = self.tokenizer(
            normalized,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        with torch.no_grad():
            logits = self.model(**encoded).logits
            probs = torch.softmax(logits, dim=-1)[0]

        k = top_k if top_k is not None else self.top_k
        k = max(1, min(k, probs.numel()))

        values, indices = torch.topk(probs, k)

        id2label = self.model.config.id2label
        candidates = []
        for idx, score in zip(indices.tolist(), values.tolist()):
            label = id2label.get(idx)
            if label is None:
                label = id2label.get(str(idx), str(idx))
            candidates.append({"idx": idx, "department": label, "confidence": float(score)})

        return {
            "input": text,
            "processed_input": normalized,
            "top1": candidates[0],
            "topk": candidates,
        }


def build_reply(prediction: Dict[str, Any]) -> str:
    top1 = prediction["top1"]
    lines = [
        f"推荐科室: {top1['department']}",
        f"置信度: {top1['confidence']:.4f}",
        "",
        "Top candidates:",
    ]
    for i, item in enumerate(prediction["topk"], start=1):
        lines.append(f"{i}. {item['department']} ({item['confidence']:.4f})")
    return "\n".join(lines)


def launch_app(classifier: DepartmentClassifier, host: str, port: int) -> None:
    with gr.Blocks(title="Department Triage Assistant") as demo:
        gr.Markdown("## Medical Department Triage Chat")
        gr.Markdown("输入症状或咨询内容，模型将返回建议科室。")

        chatbot = gr.Chatbot(label="Conversation", height=420)
        with gr.Row():
            msg = gr.Textbox(label="Input", placeholder="例如：最近咳嗽发烧胸闷，应该挂什么科？")
        with gr.Row():
            top_k = gr.Slider(minimum=1, maximum=10, step=1, value=5, label="Top-K")
        with gr.Row():
            send_btn = gr.Button("Send", variant="primary")
            clear_btn = gr.Button("Clear")

        def on_send(message: str, history: List[Dict[str, str]], k: int):
            history = history or []
            if not message.strip():
                return "", history
            try:
                prediction = classifier.predict(message, top_k=int(k))
                reply = build_reply(prediction)
            except Exception as exc:
                reply = f"Error: {exc}"

            history = history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": reply},
            ]
            return "", history

        send_btn.click(on_send, inputs=[msg, chatbot, top_k], outputs=[msg, chatbot])
        msg.submit(on_send, inputs=[msg, chatbot, top_k], outputs=[msg, chatbot])
        clear_btn.click(lambda: [], outputs=[chatbot])

    demo.launch(server_name=host, server_port=port)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple dialog UI for department prediction")
    parser.add_argument(
        "--experiment-dir",
        default="outputs/mc-bert-data-lr2e5-bs128",
        help="Experiment output folder containing run_summary/checkpoints",
    )
    parser.add_argument(
        "--model-dir",
        default="",
        help="Optional explicit model/checkpoint path; if set, skip auto best checkpoint detection",
    )
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--disable-jieba", action="store_true")
    parser.add_argument(
        "--test-text",
        default="",
        help="If set, run one prediction in CLI and exit",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.model_dir:
        model_dir = Path(args.model_dir).expanduser().resolve()
        summary: Dict[str, Any] = {}
    else:
        experiment_dir = Path(args.experiment_dir).expanduser().resolve()
        model_dir, summary = resolve_best_model_dir(experiment_dir)

    use_jieba = bool(summary.get("use_jieba", True)) and (not args.disable_jieba)

    classifier = DepartmentClassifier(
        model_dir=model_dir,
        use_jieba=use_jieba,
        max_length=args.max_length,
        top_k=args.top_k,
    )

    print(f"Loaded model from: {model_dir}")
    print(f"Use jieba preprocessing: {use_jieba}")

    if args.test_text:
        prediction = classifier.predict(args.test_text)
        print(build_reply(prediction))
        return

    launch_app(classifier, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
