import torch.nn as nn

class LoRALayer(nn.Module):
    def __init__(self, original_layer, r=4):
        super(LoRALayer, self).__init__()
        self.original_layer = original_layer
        self.lora_A = nn.Linear(original_layer.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, original_layer.out_features, bias=False)

    def forward(self, x):
        return self.original_layer(x) + self.lora_B(self.lora_A(x))
