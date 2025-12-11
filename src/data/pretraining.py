# import torch
from torch.utils.data import Dataset
from data.utils import (
    load_hf_dataset,
    add_dataset_index,
    preprocess_pretraining_instance,
)


class CompletionDataset(Dataset):
    def __init__(
        self,
        hf_args,
        template_args,
        tokenizer,
        prefix_key="prompt",
        text_key="text",
        max_length=2048,
        predict_with_generate=False,
        insert_space=False,
    ):
        super(CompletionDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = load_hf_dataset(**hf_args)
        self.data = add_dataset_index(self.data)
        # if either key does not exist in dataset, it is taken as ""
        self.prefix_key = prefix_key
        self.text_key = text_key
        self.predict_with_generate = predict_with_generate
        self.insert_space = insert_space

    def __len__(self):
        return len(self.data)

    def _process_sample(self, prefix, text_content, index=-1):
        tokenized_data = preprocess_pretraining_instance(
            self.tokenizer,
            prefix,
            text_content,
            self.max_length,
            self.predict_with_generate,
            self.insert_space,
        )
        item_dct = {
            "input_ids": tokenized_data["input_ids"],
            "labels": tokenized_data["labels"],
            "attention_mask": tokenized_data["attention_mask"],
        }
        if index != -1:
            item_dct["index"] = index
        return item_dct

    def __getitem__(self, idx):
        pref = self.data[idx].get(self.prefix_key, "")
        text_content = self.data[idx].get(self.text_key, "")
        index = self.data[idx]["index"]
        item = self._process_sample(pref, text_content, index)
        return item


# class PretrainingDataset(Dataset):
#     def __init__(
#         self, hf_args, template_args, tokenizer, text_key="text", max_length=2048
#     ):
#         super(PretrainingDataset, self).__init__()
#         self.tokenizer = tokenizer
#         self.max_length = max_length
#         self.chunks = self._chunk_raw_text(load_hf_dataset(**hf_args)[text_key])

#     def _chunk_raw_text(self, raw_text):
#         raw_text = "\n\n".join(raw_text)
#         full_token_sequence = self.tokenizer(raw_text, add_special_tokens=False)[
#             "input_ids"
#         ]
#         num_chunks = len(full_token_sequence) // self.max_length + 1
#         chunks = []
#         for i in range(num_chunks):
#             chunks.append(
#                 self.tokenizer.decode(
#                     full_token_sequence[i * self.max_length : (i + 1) * self.max_length]
#                 )
#             )
#         return chunks

#     def __len__(self):
#         return len(self.chunks)

#     def __getitem__(self, idx):
#         return preprocess_pretraining_instance(
#             self.tokenizer, "", self.chunks[idx], self.max_length
#         )

import torch

class PretrainingDataset(Dataset):
    def __init__(
        self, 
        hf_args, 
        template_args,  # 保持签名兼容性，但在此实现中未使用
        tokenizer, 
        text_key="text", 
        max_length=2048
    ):
        super(PretrainingDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_key = text_key

        # 1. 加载数据集对象（这通常是惰性的，不加载数据）
        print("Loading dataset reference...")
        dataset = load_hf_dataset(**hf_args)
        
        # 确保 Llama-2 的 tokenizer 有 eos_token
        # Llama-2-hf 的 tokenizer 应该总是有
        if tokenizer.eos_token_id is None:
            # Llama-2-hf 的 eos_token 是 '</s>'，ID 通常是 2
            tokenizer.eos_token_id = 2 
            print("Warning: tokenizer.eos_token_id was None. Manually setting to 2 (</s>).")

        # 2. 迭代、分词和拼接
        print(f"Tokenizing and concatenating all documents with key '{text_key}'... This may take time.")
        self.all_token_ids = []
        # 遍历数据集
        for example in dataset:
            text = example.get(self.text_key)
            if text:  # 确保文本不是空的
                # 分词，不添加特殊 token（我们手动添加 EOS）
                token_ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
                # 添加 EOS token 来分隔文档
                token_ids.append(self.tokenizer.eos_token_id)
                # 扩展到总列表
                self.all_token_ids.extend(token_ids)

        print(f"Total number of tokens processed: {len(self.all_token_ids)}")

        # 3. 计算总共的完整 chunk 数量
        # 我们只使用完整的 chunk，丢弃末尾不足 max_length 的部分
        self.num_chunks = len(self.all_token_ids) // self.max_length
        
        if self.num_chunks == 0:
            print(f"Warning: Total tokens ({len(self.all_token_ids)}) is less than max_length ({self.max_length}). No full chunks available.")
        
        print(f"Created {self.num_chunks} chunks of size {self.max_length}.")
        
        # 不再需要 _chunk_raw_text 和 self.chunks

    def __len__(self):
        # 返回我们能创建的 *完整* chunk 的数量
        return self.num_chunks

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.num_chunks:
            raise IndexError(f"Index {idx} is out of range for {self.num_chunks} chunks.")
            
        # 4. 直接从 token ID 列表中切片
        start_idx = idx * self.max_length
        end_idx = (idx + 1) * self.max_length
        
        # 获取 token ID 块
        input_ids = self.all_token_ids[start_idx:end_idx]
        
        # 5. 准备模型输入
        # 对于因果语言模型（Causal LM）预训练，labels 就是 input_ids
        # Trainer 或 DataCollator 会自动处理 "shift labels"
        labels = input_ids.copy()
        
        # 因为我们只使用完整的 chunk，所以 attention_mask 全是 1
        attention_mask = [1] * len(input_ids)

        # 返回 transformers.Trainer 期望的标准字典格式
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }