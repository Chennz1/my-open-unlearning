#!/bin/bash

# 强制使用本地缓存，避免联网查找数据集
export HF_DATASETS_OFFLINE=1
# 确保 datasets 库查找用户指定位置的缓存 (对应用户提供的 ~/.cache/huggingface/hub/...)
export HF_HOME=$HOME/.cache/huggingface

# 自动获取一个空闲端口 (保留原脚本逻辑)
export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"

# 定义要评测的模型名称 (需要对应 configs/model/ 下的文件名，不含 .yaml)
models=(
    "Llama-2-7b-chat-hf"
)

# 指定模型权重的具体路径
# 这里使用了你脚本中示例的路径，可以根据需要修改为变量或循环
path="/cnz/data/project/my-open-unlearning/saves/unlearn/tofu_Llama-2-7b-chat-hf_forget10_NPO_beta0.2_gamma0.2_epoch5"

for model in "${models[@]}"; do
    task_name="mmlu_${model}" # 给任务起个名字，方便日志区分

    echo "Running MMLU evaluation for $model at $path"

    # 核心评测命令
    CUDA_VISIBLE_DEVICES=0 python src/eval.py \
    eval=lm_eval \
    model=${model} \
    task_name=${task_name} \
    model.model_args.pretrained_model_name_or_path=${path} \
    model.tokenizer_args.pretrained_model_name_or_path="/cnz/data/ms-home/Llama-2-7b-chat-hf" \
    paths.output_dir=${path}/mmlu_evals
done