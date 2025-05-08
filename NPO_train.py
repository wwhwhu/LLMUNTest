import copy
import torch
from models.transformer_model import TransformerModel
from utils.data_loader import get_wmdpbio_dataloader
from utils.lora import LoRALayer
import torch.nn.functional as F
import pandas as pd
import os

def compute_npo_loss(model, initial_model, input_ids, beta=0.1):
    with torch.no_grad():
        initial_outputs = initial_model(input_ids=input_ids)
        initial_logits = initial_outputs.logits

    outputs = model(input_ids=input_ids)
    current_logits = outputs.logits

    # Shift for token prediction
    current_logits = current_logits[:, :-1, :].contiguous()
    initial_logits = initial_logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()

    # Get log probs
    current_log_probs = F.log_softmax(current_logits, dim=-1)
    initial_log_probs = F.log_softmax(initial_logits, dim=-1)

    # Select ground truth token log probs
    current_selected = current_log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
    initial_selected = initial_log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)

    # DPO loss: -log(σ[β * (π(x) / π_ref(x))])
    diff = current_selected / initial_selected
    loss = -torch.log(torch.sigmoid(beta * diff))

    return loss.mean()

def apply_lora_to_gpt2(model, r=4):
    for block in model.model.transformer.h:
        fc_layer = block.mlp.c_fc
        block.mlp.c_fc = LoRALayer(fc_layer, r)
    return model

def npo_train_on_forget_set(
    epochs=5,
    lr=1e-3,
    batch_size=32,
    lora_r=4,
    beta=1.0,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    tokenizer = TransformerModel("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # 仅加载遗忘集
    support_loader = get_wmdpbio_dataloader(batch_size=batch_size, tokenizer=tokenizer)

    # 初始化模型和优化器
    model = TransformerModel().to(device)
    model = apply_lora_to_gpt2(model, r=lora_r)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # Freeze a reference copy for DPO
    initial_model = copy.deepcopy(model).eval()
    for param in initial_model.parameters():
        param.requires_grad = False
    loss_history = []

    os.makedirs("res/npo", exist_ok=True)

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        model.train()
        for step, batch in enumerate(support_loader):
            input_ids = batch["input_ids"].to(device)

            loss = compute_npo_loss(model, initial_model, input_ids, beta=beta)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Step {step}: NPO Loss = {loss.item():.4f}")
            loss_history.append(loss.item())

        # 保存每轮的模型参数
        torch.save(model.state_dict(), f"res/npo/model_epoch_{epoch+1}.pth")

    # 保存loss曲线
    loss_df = pd.DataFrame({"npo_loss": loss_history})
    loss_df.to_csv("res/npo/loss.csv", index=False)
    print("NPO Training complete. Models and losses saved under res/npo/")

if __name__ == "__main__":
    npo_train_on_forget_set(
        epochs=10,
        lr=1e-3,
        batch_size=32,
        lora_r=4,
        beta=0.1,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
