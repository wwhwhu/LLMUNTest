import torch
from models.transformer_model import TransformerModel
from utils.data_loader import get_wikitext2_dataloader, get_wmdpbio_dataloader
from utils.lora import LoRALayer
import pandas as pd
import os

def compute_loss(model, input_ids, labels):
    outputs = model(input_ids=input_ids, labels=labels)
    return outputs.loss

def apply_lora_to_gpt2(model, r=4):
    for block in model.model.transformer.h:
        fc_layer = block.mlp.c_fc
        block.mlp.c_fc = LoRALayer(fc_layer, r)
    return model

def ga_gd_joint_train(
    epochs=5,
    lr=1e-4,
    batch_size=32,
    lora_r=4,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    tokenizer = TransformerModel("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Load both retain (query) and forget (support) dataloaders
    retain_loader = get_wikitext2_dataloader(batch_size=batch_size, tokenizer=tokenizer)
    forget_loader = get_wmdpbio_dataloader(batch_size=batch_size, tokenizer=tokenizer)

    # Initialize model
    model = TransformerModel().to(device)
    model = apply_lora_to_gpt2(model, r=lora_r)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loss_history = {"ga_loss": [], "gd_loss": []}

    os.makedirs("res/ga_gd", exist_ok=True)

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        model.train()

        forget_iter = iter(forget_loader)
        retain_iter = iter(retain_loader)

        for step in range(len(retain_loader)):
            try:
                forget_batch = next(forget_iter)
            except StopIteration:
                forget_iter = iter(forget_loader)
                forget_batch = next(forget_iter)

            try:
                retain_batch = next(retain_iter)
            except StopIteration:
                retain_iter = iter(retain_loader)
                retain_batch = next(retain_iter)

            # === 1. Gradient Ascent on forget batch ===
            input_ids_forget = forget_batch["input_ids"].to(device)
            labels_forget = input_ids_forget.clone()
            loss_forget = compute_loss(model, input_ids_forget, labels_forget)

            # === 2. Gradient Descent on retain batch ===
            input_ids_retain = retain_batch["input_ids"].to(device)
            labels_retain = input_ids_retain.clone()
            loss_retain = compute_loss(model, input_ids_retain, labels_retain)
            loss_all = loss_retain - loss_forget
            optimizer.zero_grad()
            loss_all.backward()
            optimizer.step()

            # Logging
            print(f"Step {step}: GA Loss = {loss_forget.item():.4f}, GD Loss = {loss_retain.item():.4f}")
            loss_history["ga_loss"].append(loss_forget.item())
            loss_history["gd_loss"].append(loss_retain.item())

        # 保存每轮的模型
        torch.save(model.state_dict(), f"res/ga_gd/model_epoch_{epoch+1}.pth")

    # 保存loss曲线
    pd.DataFrame(loss_history).to_csv("res/ga_gd/loss.csv", index=False)
    print("GA+GD Training complete. Results saved under res/ga_gd/")

if __name__ == "__main__":
    ga_gd_joint_train(
        epochs=10,
        lr=1e-4,
        batch_size=32,
        lora_r=4,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
