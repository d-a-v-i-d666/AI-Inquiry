#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from peft import PeftModel
from pydantic import BaseModel, Field
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
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


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


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


def build_system_prompt(categories: List[str]) -> str:
    category_list = "、".join(categories)
    return (
        "你是医疗分诊助手。请根据用户描述判断最合适的科室。\n"
        "仅输出严格 JSON，不要输出多余文本或 Markdown。\n"
        "JSON schema: {\"department\": <科室名称>, \"confidence\": <0-1>, \"reason\": <简要理由>}\n"
        f"可选科室列表: {category_list}"
    )


def build_prompt_text(tokenizer, system_prompt: str, user_text: str) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    return f"System: {system_prompt}\nUser: {user_text}\nAssistant:"


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    snippet = text[start : end + 1]
    try:
        return json.loads(snippet)
    except json.JSONDecodeError:
        return None


def extract_json_snippet(text: str) -> Optional[str]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1]


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
        device: Optional[str] = None,
    ):
        self.model_dir = model_dir
        self.use_jieba = use_jieba
        self.max_length = max_length
        self.top_k = top_k

        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(model_dir),
            local_files_only=True,
        )
        self.model.to(self.device)
        self.model.eval()

    def predict(
        self,
        text: str,
        top_k: Optional[int] = None,
        disable_jieba: bool = False,
    ) -> Dict[str, Any]:
        normalized = segment_text(text, use_jieba=self.use_jieba and not disable_jieba)
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


class QwenTriageModel:
    def __init__(
        self,
        model_dir: Path,
        adapter_dir: Path,
        use_jieba: bool,
        max_length: int,
        max_new_tokens: int,
    ):
        self.model_dir = model_dir
        self.adapter_dir = adapter_dir
        self.use_jieba = use_jieba
        self.max_length = max_length
        self.max_new_tokens = max_new_tokens
        self.system_prompt = build_system_prompt(SOURCE_CATEGORIES)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float16,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_dir),
            local_files_only=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            local_files_only=True,
            quantization_config=bnb_config,
            device_map="auto",
        )
        self.model = PeftModel.from_pretrained(base_model, str(adapter_dir), is_trainable=False)
        self.model.eval()

    def predict(
        self,
        text: str,
        max_new_tokens: Optional[int] = None,
        disable_jieba: bool = False,
    ) -> Dict[str, Any]:
        normalized = segment_text(text, use_jieba=self.use_jieba and not disable_jieba)
        if not normalized:
            raise ValueError("Input text is empty")

        prompt_text = build_prompt_text(self.tokenizer, self.system_prompt, normalized)
        encoded = self.tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=self.max_length)
        encoded = {k: v.to(self.model.device) for k, v in encoded.items()}

        with torch.no_grad():
            output_ids = self.model.generate(
                **encoded,
                max_new_tokens=max_new_tokens or self.max_new_tokens,
                do_sample=False,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        generated_ids = output_ids[0][encoded["input_ids"].shape[1] :]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        generated_text = generated_text.replace("\ufffd", "").strip()
        snippet = extract_json_snippet(generated_text)
        payload = None
        if snippet:
            try:
                payload = json.loads(snippet)
            except json.JSONDecodeError:
                payload = None

        department = None
        confidence = None
        reason = None
        parsed = False
        if isinstance(payload, dict):
            department = payload.get("department") or payload.get("科室")
            confidence = payload.get("confidence")
            reason = payload.get("reason")
            parsed = True

        raw_output = snippet if snippet else generated_text

        return {
            "input": text,
            "processed_input": normalized,
            "department": department,
            "confidence": confidence,
            "reason": reason,
            "raw_output": raw_output,
            "parsed": parsed,
        }


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1)
    top_k: int = Field(5, ge=1, le=10)
    disable_jieba: bool = False


class Candidate(BaseModel):
    idx: int
    department: str
    confidence: float


class PredictResponse(BaseModel):
    input: str
    processed_input: str
    top1: Candidate
    topk: List[Candidate]


class QwenPredictRequest(BaseModel):
    text: str = Field(..., min_length=1)
    max_new_tokens: int = Field(128, ge=1, le=512)
    disable_jieba: bool = False


class QwenPredictResponse(BaseModel):
    input: str
    processed_input: str
    department: Optional[str] = None
    confidence: Optional[float] = None
    reason: Optional[str] = None
    raw_output: str
    parsed: bool


class HealthResponse(BaseModel):
    status: str


def create_app(
    classifier: DepartmentClassifier,
    qwen_service: Optional[QwenTriageModel],
) -> FastAPI:
    app = FastAPI(title="Department Triage API", version="1.0.0")

    @app.get("/healthz", response_model=HealthResponse)
    def healthz() -> HealthResponse:
        return HealthResponse(status="ok")

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest) -> PredictResponse:
        if not req.text.strip():
            raise HTTPException(status_code=400, detail="text is empty")
        try:
            result = classifier.predict(
                req.text,
                top_k=req.top_k,
                disable_jieba=req.disable_jieba,
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return PredictResponse(**result)

    @app.post("/qwen/predict", response_model=QwenPredictResponse)
    def predict_qwen(req: QwenPredictRequest) -> QwenPredictResponse:
        if qwen_service is None:
            raise HTTPException(status_code=503, detail="Qwen service is not configured")
        if not req.text.strip():
            raise HTTPException(status_code=400, detail="text is empty")
        try:
            result = qwen_service.predict(
                req.text,
                max_new_tokens=req.max_new_tokens,
                disable_jieba=req.disable_jieba,
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return QwenPredictResponse(**result)

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Department triage inference API")
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
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--disable-jieba", action="store_true")
    parser.add_argument("--device", default="")
    parser.add_argument("--qwen-model-dir", default="")
    parser.add_argument("--qwen-adapter-dir", default="")
    parser.add_argument("--qwen-max-length", type=int, default=1024)
    parser.add_argument("--qwen-max-new-tokens", type=int, default=128)
    parser.add_argument("--qwen-disable-jieba", action="store_true")
    return parser.parse_args()


def build_classifier(args: argparse.Namespace) -> DepartmentClassifier:
    if args.model_dir:
        model_dir = Path(args.model_dir).expanduser().resolve()
        summary: Dict[str, Any] = {}
    else:
        experiment_dir = Path(args.experiment_dir).expanduser().resolve()
        model_dir, summary = resolve_best_model_dir(experiment_dir)

    use_jieba = bool(summary.get("use_jieba", True)) and (not args.disable_jieba)

    device = args.device.strip() or None
    return DepartmentClassifier(
        model_dir=model_dir,
        use_jieba=use_jieba,
        max_length=args.max_length,
        top_k=args.top_k,
        device=device,
    )


def build_qwen_service(args: argparse.Namespace) -> Optional[QwenTriageModel]:
    if not args.qwen_model_dir or not args.qwen_adapter_dir:
        return None

    model_dir = Path(args.qwen_model_dir).expanduser().resolve()
    adapter_dir = Path(args.qwen_adapter_dir).expanduser().resolve()
    use_jieba = not args.qwen_disable_jieba
    return QwenTriageModel(
        model_dir=model_dir,
        adapter_dir=adapter_dir,
        use_jieba=use_jieba,
        max_length=args.qwen_max_length,
        max_new_tokens=args.qwen_max_new_tokens,
    )


def main() -> None:
    args = parse_args()
    classifier = build_classifier(args)
    qwen_service = build_qwen_service(args)
    app = create_app(classifier, qwen_service)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
