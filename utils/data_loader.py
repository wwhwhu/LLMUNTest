from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
import torch

def tokenize_function(example, tokenizer, block_size=512):
    # 拼接并分块
    tokens = tokenizer(example["text"], return_special_tokens_mask=False, truncation=False)["input_ids"]
    result = []
    for i in range(0, len(tokens) - block_size + 1, block_size):
        result.append({"input_ids": tokens[i:i + block_size]})
    return result

class LanguageModelingDataset(torch.utils.data.Dataset):
    def __init__(self, token_blocks):
        self.blocks = token_blocks

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, idx):
        return {"input_ids": torch.tensor(self.blocks[idx]["input_ids"], dtype=torch.long)}

def get_wikitext2_dataloader(batch_size=2, tokenizer=None, block_size=512):
    if tokenizer is None:
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_from_disk("data/wikitext-103-v1")["train"]
    token_blocks = []
    for example in dataset:
        token_blocks.extend(tokenize_function(example, tokenizer, block_size))
    dataset = LanguageModelingDataset(token_blocks)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def get_wmdpbio_dataloader(batch_size=2, tokenizer=None, block_size=512):
    if tokenizer is None:
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_from_disk("data/wmdp-bio")["train"]
    token_blocks = []
    for example in dataset:
        token_blocks.extend(tokenize_function(example, tokenizer, block_size))
    dataset = LanguageModelingDataset(token_blocks)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
