from trainer.unlearn.base import UnlearnTrainer
import copy
import torch
import torch.nn as nn
from trainer.utils import compute_kl_divergence

from transformers import AutoModelForCausalLM

import torch.nn.functional as F

def _kl_on_valid_positions(student_out, teacher_out, inputs):
    # shift to next-token prediction space
    stu = student_out.logits[..., :-1, :].contiguous()
    tea = teacher_out.logits[..., :-1, :].contiguous()

    # build a mask of valid positions
    if "labels" in inputs and inputs["labels"] is not None:
        mask = inputs["labels"][..., 1:] != -100
    else:
        mask = inputs["attention_mask"][..., 1:].bool()

    # flatten and keep only valid positions
    stu = F.log_softmax(stu, dim=-1).view(-1, stu.size(-1))[mask.view(-1)]
    tea = F.log_softmax(tea, dim=-1).view(-1, tea.size(-1))[mask.view(-1)]

    return F.kl_div(stu, tea, reduction="batchmean", log_target=True)

# inside compute_loss



class NullSpace(UnlearnTrainer):

    def __init__(self, gamma=1.0, alpha=1.0, retain_loss_type="NLL", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.retain_loss_type = retain_loss_type
        self.ref_model = AutoModelForCausalLM.from_pretrained("/home/cnz/.cache/huggingface/hub/Llama-3___2-1B-Instruct")
        self.ref_model.train()
        # if retain_loss_type == "KL":
        self.ref_model = self._prepare_deepspeed(self.ref_model)

    # def _prepare_ref_model(self, model):
    #     ref_model = copy.deepcopy(model).to(self.accelerator.device)
    #     ref_model.train()
    #     if self.is_deepspeed_enabled:
    #         ref_model = self._prepare_deepspeed(ref_model)
            
    #     else:
    #         ref_model = self.accelerator.prepare_model(ref_model, evaluation_mode=True)
    #     return ref_model

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

        forget_inputs = inputs["forget"]
        retain_inputs = inputs["retain"]

        # 1. 计算 retain_loss
        retain_loss = self.compute_retain_loss(model=model, retain_inputs=retain_inputs)

        # 2. 计算 forget_loss
        forget_outputs = model(**forget_inputs)
        with torch.no_grad():
            ref_outputs = self.ref_model(**forget_inputs)

        forget_loss = _kl_on_valid_positions(forget_outputs, ref_outputs, forget_inputs)


        if forget_loss.dim() > 0:
            forget_loss = forget_loss.mean()

        # 返回一个组合的标量损失用于日志记录
        loss = self.alpha * retain_loss + self.gamma * forget_loss
        return (loss, forget_outputs) if return_outputs else loss

