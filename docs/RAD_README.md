# RAD (Representation-based Adaptive Deletion) Unlearning Algorithm

RAD是一种基于额外模块的机器遗忘算法，通过自适应地修改模型表示来实现对特定数据的遗忘。

## 算法概述

RAD算法的核心思想是：
1. **额外适应模块**：添加额外的神经网络模块来学习如何修改模型的内部表示
2. **表示调整**：通过这些模块将需要遗忘的数据的表示推离原始表示空间
3. **平衡损失**：平衡遗忘损失、适应损失和保留损失

## 主要参数

- `alpha`: 遗忘损失权重 (默认: 1.0)
- `beta`: 适应损失权重 (默认: 0.5)  
- `gamma`: 保留损失权重 (默认: 1.0)
- `module_dim`: 适应模块维度 (默认: 128)
- `num_layers`: 适应模块层数 (默认: 2)
- `retain_loss_type`: 保留损失类型 ("NLL" 或 "KL", 默认: "NLL")
- `adaptation_steps`: 适应步骤频率 (默认: 10)

## 使用方法

### 1. 使用预定义的实验配置

#### TOFU基准测试
```bash
python run_unlearn.py --config-path configs/experiment/unlearn/tofu --config-name rad
```

#### MUSE基准测试  
```bash
python run_unlearn.py --config-path configs/experiment/unlearn/muse --config-name rad
```

#### WMDP基准测试
```bash
python run_unlearn.py --config-path configs/experiment/unlearn/wmdp --config-name rad
```

### 2. 自定义配置

你可以通过命令行覆盖参数：

```bash
python run_unlearn.py --config-path configs/experiment/unlearn/tofu --config-name rad \
    trainer.method_args.alpha=2.0 \
    trainer.method_args.beta=0.3 \
    trainer.method_args.module_dim=256
```

### 3. 直接使用trainer配置

```bash
python run_unlearn.py trainer=RAD \
    trainer.method_args.alpha=1.5 \
    trainer.method_args.retain_loss_type=KL
```

## 文件结构

- `src/trainer/unlearn/rad.py`: RAD算法实现
- `configs/trainer/RAD.yaml`: RAD trainer配置
- `configs/experiment/unlearn/tofu/rad.yaml`: TOFU实验配置
- `configs/experiment/unlearn/muse/rad.yaml`: MUSE实验配置  
- `configs/experiment/unlearn/wmdp/rad.yaml`: WMDP实验配置

## 算法特点

1. **适应性强**: 通过额外模块学习特定于数据的遗忘策略
2. **表示级操作**: 直接在表示空间进行遗忘，更加精细
3. **平衡性好**: 通过多重损失函数平衡遗忘效果和模型性能
4. **可配置**: 丰富的参数设置适应不同场景需求

## 注意事项

- 适应模块会增加模型参数量和计算开销
- 建议根据具体任务调整超参数，特别是各损失权重
- 对于安全关键应用（如WMDP），建议使用更高的alpha值和更强的适应设置
