import torch
import transformers
from typing import Dict, Sequence
from data.utils import IGNORE_INDEX


class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        padding_side: str = "right",
        index: str = None,
    ):
        self.tokenizer = tokenizer
        self.padding_side = padding_side
        self.index = index

    def get_instances_from_key(self, instances: Sequence[Dict], key: str):
        ret_instances = [instance[key] for instance in instances]
        return ret_instances

    def _pad_tokens(self, input_ids, padding_value):
        if self.padding_side == "right":
            input_ids = torch.nn.utils.rnn.pad_sequence(
                input_ids, batch_first=True, padding_value=padding_value
            )
        else:
            input_ids = torch.nn.utils.rnn.pad_sequence(
                [torch.flip(i, dims=[0]) for i in input_ids],
                batch_first=True,
                padding_value=padding_value,
            ).flip(dims=[1])
        return input_ids

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        assert isinstance(instances[0], dict)
        return_dct = {}
        if "input_ids" not in instances[0]:
            for key in instances[0].keys():
                key_instances = self.get_instances_from_key(
                    instances=instances, key=key
                )
                return_dct[key] = self(key_instances)
        else:
            input_ids = [instance["input_ids"] for instance in instances]
            input_ids = self._pad_tokens(input_ids, self.tokenizer.pad_token_id)
            attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
            return_dct.update({"input_ids": input_ids})
            return_dct.update({"attention_mask": attention_mask})
            
            # return_dct.update({"interv_positions": None})
            if "labels" in instances[0]:
                labels = [instance["labels"] for instance in instances]
                labels = self._pad_tokens(labels, IGNORE_INDEX)
                if labels.size(1) != attention_mask.size(1):
                    end_indices = (input_ids != -100).sum(dim=-1)
                    start_indices = torch.clamp(end_indices - 5, min=0)
                    sequence_length = input_ids.size(1)
                    col_indices = torch.arange(sequence_length)
                    mask_ge_start = col_indices >= start_indices.unsqueeze(1)
                    mask_lt_end = col_indices <= end_indices.unsqueeze(1)
                    interv_positions = mask_ge_start & mask_lt_end
                    # return_dct.update({"interv_positions": interv_positions})
                    
                else:
                    valid_mask = ((labels != -100) ^ (attention_mask))
                    end_indices = valid_mask.sum(dim=-1)
                    start_indices = torch.clamp(end_indices - 5, min=0)
                    sequence_length = valid_mask.size(1)
                    col_indices = torch.arange(sequence_length)
                    mask_ge_start = col_indices > start_indices.unsqueeze(1)
                    mask_lt_end = col_indices <= end_indices.unsqueeze(1)
                    interv_positions = mask_ge_start & mask_lt_end
                    # return_dct.update({"interv_positions": interv_positions})
                return_dct.update({"labels": labels})
            if self.index:
                if self.index in instances[0]:
                    return_dct.update(
                        {
                            self.index: torch.tensor(
                                [example[self.index] for example in instances]
                            )
                        }
                    )
                else:
                    raise Warning(f"{self.index} not found in dataset")
        return return_dct
