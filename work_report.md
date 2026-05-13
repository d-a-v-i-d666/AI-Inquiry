# AI-Inquiry 工作报告（阶段复盘与路线调整）

## 1. 本阶段目标
本阶段围绕“智能分诊分类”完成两条技术路线实践：
1. 基于 `dataset` 数据，在 `projects/AI-Inquiry/train.ipynb` 中完成初版训练与验证。
2. 基于 `data` 数据，在 `projects/AI-Inquiry/train.py` 中完成脚本化训练、实验对比与模型落地。
3. 基于最佳模型，快速搭建可交互的简易对话框系统。

## 2. 阶段一回顾：`train.ipynb` + `dataset`（已完成，后续弃用）

### 2.1 完成内容
- 在 `projects/AI-Inquiry/train.ipynb` 中完成端到端微调流程。
- 使用数据：`dataset/triage/label_1_cls_data/`。
- 初版任务类别数：14。

### 2.2 阶段结果（历史有效）
- 最优 `eval_loss`：0.5614（best step=2190）。
- 参考指标：`eval_accuracy=0.8146`，`eval_macro_f1=0.6338`。
- 相关产物目录：`outputs/mc-bert-label1/`。

### 2.3 弃用原因
虽然该阶段指标可观，但该数据集覆盖的分诊科室数量较少（14 类），与实际分诊场景的科室覆盖需求不匹配。为提升可用性与泛化能力，后续训练主线切换到 `data` 数据源。

## 3. 当前主线：`train.py` + `data`（重点）

### 3.1 训练方案
- 脚本：`projects/AI-Inquiry/train.py`。
- 数据：
  - `data/v2.0Train.txt`
  - `data/v2.0Val.txt`
  - `data/v2.0Test.txt`
- 任务类别：34 个科室。
- 训练流程：固定科室映射 + 文本预处理（可选 jieba）+ BERT 分类微调 + 自动保存最佳 checkpoint。

### 3.2 关键实验对比（`data` 路线）

| 实验目录 | 最优 checkpoint | Val Loss | Val Acc | Val Macro F1 | Test Loss | Test Acc | Test Macro F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `outputs/mc-bert-data-lr2e5-bs256/` | checkpoint-234 | 1.5511 | 0.6125 | 0.5793 | 1.5570 | 0.6223 | 0.5871 |
| `outputs/mc-bert-data-lr2e5-bs128/` | **checkpoint-357** | **1.5318** | **0.6135** | **0.5816** | **1.5520** | 0.6099 | 0.5778 |

说明：当前以验证损失最优作为模型选择标准，采用 `outputs/mc-bert-data-lr2e5-bs128/checkpoint-357` 作为对话系统基座模型。

## 4. 工程落地：简易对话框系统

### 4.1 交付内容
- 对话脚本：`projects/AI-Inquiry/triage_chat_app.py`。
- 功能：输入自然语言症状，输出推荐科室、置信度和 Top-K 候选。
- 模型加载：默认自动读取实验目录中的 `run_summary.json` 并加载最佳 checkpoint。

### 4.2 联调与修复
- 已完成 Gradio 消息格式兼容修复（`role/content` 格式），对话交互可正常运行。
- 可通过命令直接启动服务进行交互验证。

## 5. 结论
1. `dataset + train.ipynb` 路线完成了有效的技术验证，但因科室覆盖不足已阶段性弃用。
2. 当前主线已切换至 `data + train.py`，并形成可复用训练脚本与多组实验基线。
3. 已基于最佳结果快速搭建分诊对话框，为后续系统升级提供可运行基座。

## 6. 下一步目标与工作计划

### 6.1 总体目标
在现有简易分类模型基础上，探索“分类模型 + 大语言模型智能体”的分诊系统，验证是否可获得更好的综合效果（准确性、可解释性、交互体验）。

### 6.2 具体计划
1. 在云服务器部署可用大语言模型服务（推理 API/服务化）。
2. 基于 AgentSkill 设计分诊智能体流程：
   - 症状理解与关键信息抽取；
   - 规则/知识增强；
   - 分类模型协同决策与结果解释。
3. 构建对比评估：
   - 现有分类模型基线 vs AgentSkill 分诊系统；
   - 指标覆盖准确率、宏平均 F1、误分样例可解释性。
4. 形成下一阶段原型：
   - 一个可在线访问的分诊演示系统；
   - 一份结构化实验评估报告。

---
报告更新日期：2026-04-13
