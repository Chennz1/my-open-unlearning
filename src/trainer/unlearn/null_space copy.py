# 实验思路1：梯度正交GA

from trainer.unlearn.base import UnlearnTrainer
import copy
import torch
import torch.nn as nn
from trainer.utils import compute_kl_divergence


class NullSpace(UnlearnTrainer):

    def __init__(self, gamma=1.0, alpha=1.0, retain_loss_type="NLL", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.retain_loss_type = retain_loss_type
        self.ref_model = None
        if retain_loss_type == "KL":
            self.ref_model = self._prepare_ref_model(self.model)

    def _prepare_ref_model(self, model):
        ref_model = copy.deepcopy(model).to(self.accelerator.device)
        ref_model.eval()
        if self.is_deepspeed_enabled:
            ref_model = self._prepare_deepspeed(ref_model)
        else:
            ref_model = self.accelerator.prepare_model(ref_model, evaluation_mode=True)
        return ref_model

    def compute_retain_loss(self, model, retain_inputs):
        retain_outputs = model(**retain_inputs)
        retain_loss_val = 0.0
        if self.retain_loss_type == "NLL":
            retain_loss_val += retain_outputs.loss
        elif self.retain_loss_type == "KL":
            kl_loss, retain_outputs = compute_kl_divergence(
                self.model, self.ref_model, retain_inputs
            )
            retain_loss_val += kl_loss
        else:
            raise NotImplementedError(
                f"{self.retain_loss_type} not implemented for retain set"
            )

        # **FIX**: Ensure the returned loss is a scalar
        if retain_loss_val.dim() > 0:
            return retain_loss_val.mean()
        return retain_loss_val

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        此方法现在仅用于日志记录，不用于梯度计算。
        梯度计算的逻辑已移至 training_step。
        """
        forget_inputs = inputs["forget"]
        retain_inputs = inputs["retain"]

        with torch.no_grad():  # 仅计算值，不追踪梯度
            # 1. 计算 retain_loss
            retain_loss = self.compute_retain_loss(model=model, retain_inputs=retain_inputs)

            # 2. 计算 forget_loss
            forget_outputs = model(**forget_inputs)
            forget_loss = -forget_outputs.loss  # 梯度上升

            if forget_loss.dim() > 0:
                forget_loss = forget_loss.mean()

        # 返回一个组合的标量损失用于日志记录
        loss = self.alpha * retain_loss + self.gamma * forget_loss
        return (loss, forget_outputs) if return_outputs else loss

    def training_step(self, model: nn.Module, inputs: dict) -> torch.Tensor:
        """
        重写 training_step 以通过构建一个组合损失来处理梯度投影，
        从而与 DeepSpeed 和 Accelerate 兼容。
        此版本经过优化，可减少显存使用。
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        # --- 1. 前向传播，计算两个损失 ---
        forget_inputs = inputs["forget"]
        retain_inputs = inputs["retain"]

        # 计算 retain_loss
        retain_loss = self.compute_retain_loss(model=model, retain_inputs=retain_inputs)

        # 计算 forget_loss
        forget_outputs = model(**forget_inputs)
        forget_loss_tensor = -forget_outputs.loss  # 梯度上升
        forget_loss = forget_loss_tensor.mean() if forget_loss_tensor.dim() > 0 else forget_loss_tensor

        # --- 2. 优化梯度计算和投影 ---
        
        # 步骤 2.1: 计算 retain 梯度 (g_r) 并存储在 .grad 中
        # 使用 self.accelerator.backward 来处理混合精度和分布式训练
        self.accelerator.backward(retain_loss, retain_graph=True)

        # 步骤 2.2: 计算投影系数
        # 我们将迭代参数，同时计算点积和 g_r 的范数
        dot_product = 0.0
        retain_grad_norm_sq = 0.0
        
        # 计算 forget 梯度 (g_f) 并立即用于计算点积
        # 这避免了将整个 g_f 存储在内存中
        forget_grads_tuple = torch.autograd.grad(
            forget_loss, model.parameters(), allow_unused=True
        )

        for p, f_grad in zip(model.parameters(), forget_grads_tuple):
            if p.grad is not None and f_grad is not None:
                r_grad = p.grad
                dot_product += torch.sum(r_grad * f_grad)
                retain_grad_norm_sq += torch.sum(r_grad ** 2)

        # 步骤 2.3: 清空梯度，为最终的 backward() 做准备
        model.zero_grad()

        # 计算投影系数
        projection = dot_product / (retain_grad_norm_sq + 1e-8)
        
        # --- 3. 构建最终的组合损失 ---
        # final_loss 的梯度将是 alpha * g_r + gamma * (g_f - proj * g_r)
        final_loss = (self.alpha - self.gamma * projection.detach()) * retain_loss + self.gamma * forget_loss

        # 返回最终的损失。Trainer 将会自动处理 backward() 和 optimizer.step()
        return final_loss
