import copy

import torch
from torch.nn.utils.rnn import pad_sequence

from trainer.unlearn.base import UnlearnTrainer
from trainer.utils import compute_kl_divergence


class SKU(UnlearnTrainer):
    """Selective Knowledge negation Unlearning acquisition trainer.

    This trainer only covers the acquisition stage. Task-vector negation/export is
    intentionally handled by a separate post-processing script.
    """

    def __init__(
        self,
        gamma=1.0,
        alpha=1.0,
        beta=1.0,
        random_answer_k=1,
        random_answer_source="forget",
        retain_loss_type="KL",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.random_answer_k = max(int(random_answer_k), 1)
        self.random_answer_source = random_answer_source
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

    def _forward_inputs(self, batch):
        return {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "labels": batch["labels"],
        }

    def compute_forget_loss(self, model, forget_inputs):
        outputs = model(**forget_inputs)
        return outputs.loss, outputs

    def compute_retain_loss(self, model, retain_inputs):
        retain_outputs = model(**retain_inputs)
        if self.retain_loss_type == "NLL":
            return retain_outputs.loss, retain_outputs
        if self.retain_loss_type == "KL":
            return compute_kl_divergence(model, self.ref_model, retain_inputs)
        raise NotImplementedError(
            f"{self.retain_loss_type} not implemented for retain set"
        )

    def _get_answer_source_inputs(self, forget_inputs, retain_inputs):
        if self.random_answer_source == "forget":
            return forget_inputs
        if self.random_answer_source == "retain":
            return retain_inputs
        raise ValueError(
            f"Unsupported random_answer_source: {self.random_answer_source}. "
            "Expected one of ['forget', 'retain']."
        )

    def build_random_mismatch_batch(self, forget_inputs, retain_inputs):
        """Pair each forget prompt with a randomly permuted answer source.

        Original SKU samples random answers from the harmful / forget pool. This is
        therefore the default behavior. A retain-source variant is still exposed as
        a configurable ablation.
        """
        answer_source_inputs = self._get_answer_source_inputs(
            forget_inputs=forget_inputs,
            retain_inputs=retain_inputs,
        )
        device = forget_inputs["input_ids"].device
        batch_size = forget_inputs["input_ids"].size(0)
        source_batch_size = answer_source_inputs["input_ids"].size(0)
        if batch_size == 0 or source_batch_size == 0:
            return None

        if source_batch_size == 1:
            source_indices = torch.zeros(batch_size, dtype=torch.long, device=device)
        else:
            source_indices = torch.randint(
                low=0,
                high=source_batch_size,
                size=(batch_size,),
                device=device,
            )
            if self.random_answer_source == "forget" and source_batch_size == batch_size:
                same_mask = source_indices.eq(torch.arange(batch_size, device=device))
                source_indices[same_mask] = (source_indices[same_mask] + 1) % source_batch_size

        input_ids, labels, attention_masks = [], [], []
        pad_token_id = self.tokenizer.pad_token_id

        for idx in range(batch_size):
            forget_ids = forget_inputs["input_ids"][idx]
            forget_labels = forget_inputs["labels"][idx]
            answer_ids = answer_source_inputs["input_ids"][source_indices[idx]]
            answer_labels = answer_source_inputs["labels"][source_indices[idx]]

            forget_prompt_mask = forget_labels.eq(-100)
            answer_token_mask = answer_labels.ne(-100)

            prompt_tokens = forget_ids[forget_prompt_mask]
            answer_tokens = answer_ids[answer_token_mask]
            if prompt_tokens.numel() == 0 or answer_tokens.numel() == 0:
                continue

            mixed_input_ids = torch.cat([prompt_tokens, answer_tokens], dim=0)
            mixed_labels = torch.cat(
                [
                    torch.full_like(prompt_tokens, -100),
                    answer_tokens.clone(),
                ],
                dim=0,
            )
            mixed_attention = torch.ones_like(mixed_input_ids)

            input_ids.append(mixed_input_ids)
            labels.append(mixed_labels)
            attention_masks.append(mixed_attention)

        if not input_ids:
            return None

        padded_input_ids = pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=pad_token_id,
        )
        padded_labels = pad_sequence(
            labels,
            batch_first=True,
            padding_value=-100,
        )
        padded_attention_mask = pad_sequence(
            attention_masks,
            batch_first=True,
            padding_value=0,
        )
        return {
            "input_ids": padded_input_ids,
            "labels": padded_labels,
            "attention_mask": padded_attention_mask,
        }

    def compute_random_mismatch_loss(self, model, forget_inputs, retain_inputs):
        total_loss = 0.0
        mismatch_outputs = None
        valid_samples = 0
        for _ in range(self.random_answer_k):
            mismatch_batch = self.build_random_mismatch_batch(
                forget_inputs=forget_inputs,
                retain_inputs=retain_inputs,
            )
            if mismatch_batch is None:
                continue
            mismatch_outputs = model(**mismatch_batch)
            total_loss += mismatch_outputs.loss
            valid_samples += 1

        if mismatch_outputs is None or valid_samples == 0:
            return torch.tensor(0.0, device=forget_inputs["input_ids"].device), None
        return total_loss / valid_samples, mismatch_outputs

    def compute_loss(self, model, inputs, return_outputs=False):
        forget_inputs = self._forward_inputs(inputs["forget"])
        retain_inputs = self._forward_inputs(inputs["retain"])

        forget_loss, forget_outputs = self.compute_forget_loss(model, forget_inputs)
        random_loss, _ = self.compute_random_mismatch_loss(
            model,
            forget_inputs,
            retain_inputs,
        )
        retain_loss, _ = self.compute_retain_loss(model, retain_inputs)

        loss = (
            self.gamma * forget_loss
            + self.alpha * random_loss
            + self.beta * retain_loss
        )
        return (loss, forget_outputs) if return_outputs else loss
