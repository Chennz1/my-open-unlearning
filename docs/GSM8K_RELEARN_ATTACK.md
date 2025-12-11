# GSM-8K Fine-tuning for Relearn Attack Study

本目录包含用于研究 relearn attack 影响的 GSM-8K 数据集微调脚本和评估工具。

## 概述

Relearn attack 是一种针对机器遗忘(machine unlearning)的攻击方法,通过在遗忘后重新微调模型来恢复被遗忘的知识。本项目使用 GSM-8K 数学推理数据集来研究这种攻击的有效性。

## 文件结构

```
configs/
├── data/datasets/
│   ├── GSM8K_train.yaml          # GSM-8K 训练集配置
│   └── GSM8K_test.yaml           # GSM-8K 测试集配置
├── experiment/finetune/gsm8k/
│   └── default.yaml              # GSM-8K 微调实验配置
└── eval/
    └── gsm8k.yaml                # GSM-8K 评估配置

src/evals/
└── gsm8k.py                      # GSM-8K 评估器实现

scripts/
├── gsm8k_finetune_relearn.sh     # 基础微调脚本
├── gsm8k_eval.sh                 # 评估脚本
└── gsm8k_relearn_attack_pipeline.sh  # 完整的三阶段 relearn attack 研究流程
```

## 快速开始

### 1. 基础微调 (仅在 GSM-8K 上微调)

```bash
bash scripts/gsm8k_finetune_relearn.sh
```

这将在 GSM-8K 训练集上微调指定的模型,模型保存在:
- `saves/finetune/gsm8k_{model}_relearn/`

### 2. 评估微调后的模型

```bash
bash scripts/gsm8k_eval.sh
```

评估结果保存在:
- `saves/eval/gsm8k_{model}_relearn/GSM8K_EVAL.json` (详细结果)
- `saves/eval/gsm8k_{model}_relearn/GSM8K_SUMMARY.json` (汇总指标)

### 3. 完整的 Relearn Attack 研究流程

```bash
bash scripts/gsm8k_relearn_attack_pipeline.sh
```

这个脚本执行完整的三阶段流程:
- **Stage 1**: 在 GSM-8K 上微调模型
- **Stage 2**: 使用遗忘方法(如 Gradient Ascent, NPO)遗忘特定知识
- **Stage 3**: Relearn attack - 在遗忘后重新在 GSM-8K 上微调

## 配置说明

### 数据集配置 (`configs/data/datasets/GSM8K_*.yaml`)

```yaml
GSM8K_train:
  handler: QADataset
  args:
    hf_args:
      path: "openai/gsm8k"      # HuggingFace 数据集路径
      name: "main"               # 数据集子集
      split: "train"             # 数据划分
    question_key: "question"     # 问题字段名
    answer_key: "answer"         # 答案字段名
    max_length: 512              # 最大序列长度
```

### 微调配置 (`configs/experiment/finetune/gsm8k/default.yaml`)

```yaml
mode: finetune
trainer:
  args:
    learning_rate: 2e-5          # 学习率
    weight_decay: 0.01           # 权重衰减
    warmup_epochs: 0.5           # 预热轮数
    num_train_epochs: 3          # 训练轮数
    evaluation_strategy: "steps" # 评估策略
    eval_steps: 500              # 每 500 步评估一次
    save_strategy: "epoch"       # 每个 epoch 保存一次
    logging_steps: 50            # 日志记录频率
```

### 评估配置 (`configs/eval/gsm8k.yaml`)

```yaml
gsm8k:
  handler: GSM8KEvaluator
  max_new_tokens: 512            # 生成的最大 token 数
  batch_size: 4                  # 批次大小
  subset: "test"                 # 使用测试集评估
  num_samples: null              # null = 评估所有样本
```

## 自定义配置

### 修改模型

编辑脚本中的 `models` 数组:

```bash
models=(
    "Llama-3.2-1B-Instruct"
    "Llama-3.2-3B-Instruct"
    "Llama-3.1-8B-Instruct"
    "phi-1_5"
)
```

### 修改遗忘方法

在 `gsm8k_relearn_attack_pipeline.sh` 中修改 `unlearn_methods`:

```bash
unlearn_methods=(
    "GradientAscent"
    "NPO"
    "GradientDifference"
)
```

### 调整训练超参数

可以通过命令行参数覆盖配置:

```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
    --config_file configs/accelerate/default_config.yaml \
    src/train.py experiment=finetune/gsm8k/default.yaml \
    trainer.args.learning_rate=1e-5 \
    trainer.args.num_train_epochs=5 \
    trainer.args.per_device_train_batch_size=8
```

## 实验结果分析

### 评估指标

GSM-8K 评估器会计算以下指标:
- **Accuracy**: 答案准确率
- **Correct/Total**: 正确答案数 / 总问题数

### 结果文件

1. **详细结果** (`GSM8K_EVAL.json`):
   - 包含每个样本的问题、预测、正确答案
   - 提取的数值答案
   - 是否正确的标记

2. **汇总结果** (`GSM8K_SUMMARY.json`):
   ```json
   {
       "gsm8k_accuracy": 0.75,
       "gsm8k_correct": 987,
       "gsm8k_total": 1319
   }
   ```

### 比较三个阶段的性能

```bash
# Stage 1: 初始微调
cat saves/eval/gsm8k_Llama-3.2-1B-Instruct_stage1/GSM8K_SUMMARY.json

# Stage 2: 遗忘后
cat saves/eval/gsm8k_Llama-3.2-1B-Instruct_GradientAscent_stage2_unlearn/GSM8K_SUMMARY.json

# Stage 3: Relearn attack 后
cat saves/eval/gsm8k_Llama-3.2-1B-Instruct_GradientAscent_stage3_relearn/GSM8K_SUMMARY.json
```

## 答案提取机制

GSM-8K 评估器使用两种方法提取数值答案:

1. **标准格式**: `#### {answer}` (GSM-8K 的标准答案格式)
2. **后备方案**: 提取文本中的最后一个数字

答案比较时会:
- 移除千位分隔符
- 转换为浮点数
- 使用容差 1e-3 进行比较

## 注意事项

1. **GPU 内存**: 根据模型大小调整 `per_device_train_batch_size` 和 `gradient_accumulation_steps`
2. **数据集下载**: 首次运行会从 HuggingFace 下载 GSM-8K 数据集
3. **生成长度**: GSM-8K 需要较长的推理链,建议 `max_new_tokens >= 512`
4. **评估时间**: 完整评估约 1319 个样本,可能需要较长时间

## 研究问题

使用这些脚本可以研究:

1. **Relearn attack 的有效性**:
   - 遗忘后重新学习能恢复多少性能?
   - 不同遗忘方法对 relearn attack 的鲁棒性如何?

2. **遗忘的深度**:
   - 模型是真正遗忘了知识,还是只是抑制了输出?
   - 遗忘后的模型内部表示发生了什么变化?

3. **微调策略**:
   - 不同的学习率和训练轮数如何影响 relearn 效果?
   - 是否存在最优的 relearn 策略?

## 扩展

### 添加新的评估指标

在 `src/evals/gsm8k.py` 中添加新的评估函数,例如:
- 推理步骤数量
- 中间步骤的正确性
- 数学表达式的有效性

### 使用其他数学推理数据集

类似地创建配置文件:
- MATH 数据集
- AQuA-RAT
- MultiArith

## 许可证

遵循项目主许可证。

## 引用

如果使用 GSM-8K 数据集,请引用:
```
@article{cobbe2021training,
  title={Training verifiers to solve math word problems},
  author={Cobbe, Karl and Kosaraju, Vineet and Bavarian, Mohammad and Chen, Mark and Jun, Heewoo and Kaiser, Lukasz and Plappert, Matthias and Tworek, Jerry and Hilton, Jacob and Nakano, Reiichiro and others},
  journal={arXiv preprint arXiv:2110.14168},
  year={2021}
}
```
