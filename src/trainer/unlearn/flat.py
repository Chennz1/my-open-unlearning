import torch
import torch.nn.functional as F
from torch import nn

from trainer.unlearn.base import UnlearnTrainer


class TokenProbLoss(nn.Module):
    """Compute token-wise negative log-probability without reducing.

    FLAT optimizes a variational f-divergence between the forget answer and a
    template answer for the same question. We therefore need per-token losses so
    that the divergence is computed from the answer span only.
    """

    def __init__(self):
        super().__init__()
        self.loss_fn = nn.NLLLoss(reduction="none", ignore_index=-100)

    def forward(self, logits, labels):
        log_probs = F.log_softmax(logits, dim=-1)
        return self.loss_fn(log_probs, labels)


def get_flat_divergence(forget_score, template_score, divergence="Total-Variation"):
    """Variational f-divergence objective used by FLAT.

    This follows the public FLAT implementation structure, but uses stable
    log-probabilities for the sequence scores.
    """

    if divergence == "KL":
        activation = lambda x: -torch.mean(x)
        conjugate = lambda x: -torch.mean(torch.exp(x - 1.0))
    elif divergence == "Reverse-KL":
        activation = lambda x: -torch.mean(-torch.exp(x))
        conjugate = lambda x: -torch.mean(-1.0 - x)
    elif divergence == "Jeffrey":
        activation = lambda x: -torch.mean(x)
        conjugate = lambda x: -torch.mean(
            x + (x * x) / 4.0 + (x * x * x) / 16.0
        )
    elif divergence == "Squared-Hellinger":
        activation = lambda x: -torch.mean(1.0 - torch.exp(x))
        conjugate = lambda x: -torch.mean((1.0 - torch.exp(x)) / torch.exp(x))
    elif divergence == "Pearson":
        activation = lambda x: -torch.mean(x)
        conjugate = lambda x: -torch.mean((x * x) / 4.0 + x)
    elif divergence == "Neyman":
        activation = lambda x: -torch.mean(1.0 - torch.exp(x))
        conjugate = lambda x: -torch.mean(2.0 - 2.0 * torch.sqrt(1.0 - x))
    elif divergence == "Jenson-Shannon":
        log2 = torch.log(torch.tensor(2.0, device=forget_score.device))
        activation = (
            lambda x: -torch.mean(-torch.log1p(torch.exp(-x))) - log2
        )
        conjugate = (
            lambda x: -torch.mean(x + torch.log1p(torch.exp(-x))) + log2
        )
    elif divergence == "Total-Variation":
        activation = lambda x: -torch.mean(torch.tanh(x) / 2.0)
        conjugate = lambda x: -torch.mean(torch.tanh(x) / 2.0)
    else:
        raise NotImplementedError(f"Unsupported FLAT divergence: {divergence}")

    # Match the public FLAT code path: regular term on the template answer,
    # peer term on the forget answer.
    regular_term = activation(-template_score)
    peer_term = conjugate(-forget_score)
    return regular_term - peer_term


class FLAT(UnlearnTrainer):
    """FLAT unlearning trainer.

    FLAT uses only forget questions during optimization. For each forget example,
    we compare the model score on:
    - the original forget answer
    - a template answer for the same question (IDK by default on TOFU)

    The final objective is the variational f-divergence between those two answer
    distributions. Retain data may still be present in the training dataset for
    compatibility with the open-unlearning pipeline, but it is not used here.
    """

    def __init__(self, divergence="Total-Variation", template_weight=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.divergence = divergence
        self.template_weight = template_weight
        self.token_prob_loss = TokenProbLoss()

    def _forward_inputs(self, batch):
        return {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "labels": batch["labels"],
        }

    def _sequence_token_prob(self, outputs, labels):
        """Return per-example answer scores from token log-probabilities.

        FLAT only uses the supervised target span, i.e. tokens whose labels are
        not masked with -100.
        """

        shift_logits = outputs.logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        token_nll = self.token_prob_loss(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        ).view_as(shift_labels)

        valid_mask = shift_labels.ne(-100)
        token_nll = token_nll * valid_mask
        # Convert NLL to log-probability-like sequence score.
        return -token_nll.sum(dim=-1)

    def compute_loss(self, model, inputs, return_outputs=False):
        if "forget" not in inputs:
            raise ValueError("FLAT requires a forget batch.")

        forget_batch = inputs["forget"]
        if "original" not in forget_batch or "alternate" not in forget_batch:
            raise ValueError(
                "FLAT expects forget samples with 'original' and 'alternate' views. "
                "Use QAwithIdkDataset or QAwithAlternateDataset for the forget split."
            )

        forget_inputs = self._forward_inputs(forget_batch["original"])
        template_inputs = self._forward_inputs(forget_batch["alternate"])

        forget_outputs = model(**forget_inputs)
        template_outputs = model(**template_inputs)

        forget_score = self._sequence_token_prob(
            forget_outputs, forget_inputs["labels"]
        )
        template_score = self._sequence_token_prob(
            template_outputs, template_inputs["labels"]
        )

        loss = get_flat_divergence(
            forget_score=forget_score,
            template_score=self.template_weight * template_score,
            divergence=self.divergence,
        )
        return (loss, forget_outputs) if return_outputs else loss
