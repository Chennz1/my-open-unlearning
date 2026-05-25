# IA-TVC Model Merge 改进方案

## 背景问题

当前 merge 脚本主要使用固定形式：

```text
theta_u = theta_target - alpha * tau_forget + beta * tau_retain
```

其中 `alpha` 和 `beta` 通常由网格搜索或手工设定。根据 `a.md` 的分析，问题不在于 task-vector arithmetic 本身，而在于 `beta=1` 隐含了很强的假设：随机 retain vector 的方向、尺度和函数影响都能自然补偿 forget vector 被减掉的 retain-relevant 成分。仅做 same-size retain sampling 不能保证这个假设成立。

## 改进思路

把 compensation 从“加一个 retain vector”改成“在 retain task-vector 子空间里投影 forget vector”。设：

```text
tau_f = theta_forget - theta_base
T_R = [tau_r^1, ..., tau_r^K]
tau_r^k = theta_retain^k - theta_base
```

求解：

```text
w* = argmin_w ||T_R w - alpha * tau_f||_R^2
     + gamma ||T_R w||_F^2
     + lambda ||w||_2^2
```

最终：

```text
theta_u = theta_target - alpha * tau_f + T_R w*
```

实现上先提供一个不依赖数据 loader 的轻量版本：`R` 使用参数空间 identity metric，`F` 可选 `rank1` forget-direction penalty，用 `tau_f tau_f^T / ||tau_f||^2` 近似防止补偿重新沿 forget 方向激活。`K=1` 时，该方法退化为自动求 `beta*` 的 TVC；`K>1` 时可使用多个 retain shard model 构造 retain 子空间。

## 实现文件

- `scripts/ia_tvc_merge.py`：通用 IA/RSP-TVC merge 脚本。
- `scripts/search_hyperparams_ia_tvc_tofu_7b_forget10.sh`：复用原版 forget/sretain 训练产物，执行 IA-TVC merge search，并调用 `scripts/tofu_diy_eval.sh` 评估。
- `scripts/run_ia_tvc_pipeline_tofu_7b_forget10.sh`：默认只跑 search；设置 `RUN_TRAIN=1` 时才先调用原版训练脚本。
- `scripts/ia_tvc_finetune_sretain_tofu_7b.sh`：可选 shard ablation；仅在需要 `K>1` 个小 retain vectors 时使用。

## 推荐实验

1. 先用 `K=1` 替代现有 `beta` 网格搜索，比较 `beta*` 与手工最优值。
2. 再训练 2-4 个 retain shard model，用 `--retain_paths` 一次传入多个路径，验证 retain subspace 是否比单 retain vector 稳定。
3. 对比 `--forget_metric none`、`identity`、`rank1`，观察 forget quality 和 model utility 的权衡。
4. 默认只在 MLP projection 层 merge；若 retain utility 不足，再尝试 `--filter_type attn` 或 `all_linear`。
