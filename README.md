# AI-Inquiry

基于 MC-BERT 的分诊分类基线，面向后续与 Qwen3.6 结合的专家分诊系统。

## Features

- TXT → CSV 预处理脚本，统一 label_name/label_id
- MC-BERT CSV 微调脚本，输出评估指标与 loss 曲线
- Qwen3.6 QLoRA 微调脚本，固定 JSON 输出格式
- 34 个科室分类标签映射与一致化预处理
- Gradio 交互式分诊对话界面，支持 Top-K 候选

## Project Structure

- data/: v2.0Train/Val/Test 原始 TXT（每行: 科室\t文本）
- data_csv/: CSV 训练/验证/测试文件
- dataset/: 早期实验数据集
- models/: 基座与微调后的模型
- outputs/: 训练日志、checkpoint、评估产物
- prepare_data_csv.py: TXT → CSV 预处理脚本
- train_mc_bert.py: MC-BERT CSV 训练脚本
- train_qwen.py: Qwen3.6 QLoRA 训练脚本
- train.py: 旧版 TXT 训练脚本（保留）
- triage_chat_app.py: 分诊对话界面与 CLI 预测

## Quickstart

### Install

```bash
pip install -U torch transformers datasets pandas numpy scikit-learn matplotlib huggingface_hub gradio jieba peft bitsandbytes accelerate
```

如果不想使用 jieba 分词，可在训练/推理时加 `--disable-jieba`。

### Prepare CSV

```bash
python prepare_data_csv.py \
  --workspace-dir . \
  --input-dir data \
  --output-dir data_csv
```

### Train (MC-BERT)

```bash
python train_mc_bert.py \
  --workspace-dir . \
  --train-csv data_csv/train.csv \
  --val-csv data_csv/val.csv \
  --test-csv data_csv/test.csv \
  --output-dir outputs/mc-bert-csv \
  --finetuned-dir models/mc-bert-csv
```

### Train (Qwen3.6 QLoRA)

```bash
python train_qwen.py \
  --workspace-dir . \
  --train-csv data_csv/train.csv \
  --val-csv data_csv/val.csv \
  --test-csv data_csv/test.csv \
  --output-dir outputs/qwen3.6-27b-qlora \
  --adapter-dir models/qwen3.6-27b-qlora \
  --trust-remote-code \
  --run-eval
```

### Inference

CLI 预测：

```bash
python triage_chat_app.py \
  --experiment-dir outputs/mc-bert-data-lr2e5-bs128 \
  --test-text "最近咳嗽发烧胸闷，应该挂什么科？"
```

启动 Gradio：

```bash
python triage_chat_app.py \
  --experiment-dir outputs/mc-bert-data-lr2e5-bs128 \
  --host 0.0.0.0 \
  --port 7860
```

## Data Format

### CSV（训练主用）

字段：`text`、`label_name`、`label_id`。

### TXT（旧版）

每行一条样本，使用 TAB 分隔：

```
科室名称\t文本内容
```

## Notes

- 默认基座模型为 `freedomking/mc-bert`，首次训练会自动下载到 models/。
- Qwen3.6 训练需要 4-bit 量化依赖（bitsandbytes）和足够显存；必要时调小 `--max-length` 或梯度累积。
- .gitignore 已忽略 data/、data_csv/、dataset/、models/、outputs/。如需提交这些目录，请自行调整。

## License

MIT License. See LICENSE.
