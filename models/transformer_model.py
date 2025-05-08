import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel

class TransformerModel(nn.Module):
    def __init__(self, model_name='gpt2'):
        super(TransformerModel, self).__init__()
        self.model = GPT2LMHeadModel.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
