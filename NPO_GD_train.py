import torch
import torch.nn.functional as F
from models.transformer_model import TransformerModel
from utils.data_loader import get_wikitext2_dataloader, get_wmdpbio_dataloader
from utils.lora import LoRALayer
import pandas as pd
import os
import copy

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

def compute_ce_loss(model, input_ids):
    outputs = model(input_ids=input_ids, labels=input_ids)
    return outputs.loss

def apply_lora_to_gpt2(model, r=4):
    for block in model.model.transformer.h:
        fc_layer = block.mlp.c_fc
        block.mlp.c_fc = LoRALayer(fc_layer, r)
    return model

def dpo_gd_joint_train(
    epochs=5,
    lr=1e-4,
    batch_size=32,
    lora_r=4,
    beta=0.1,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    tokenizer = TransformerModel("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Load datasets
    retain_loader = get_wikitext2_dataloader(batch_size=batch_size, tokenizer=tokenizer)
    forget_loader = get_wmdpbio_dataloader(batch_size=batch_size, tokenizer=tokenizer)

    # Model initialization
    model = TransformerModel().to(device)
    model = apply_lora_to_gpt2(model, r=lora_r)

    # Freeze a reference copy for DPO
    initial_model = copy.deepcopy(model).eval()
    for param in initial_model.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_history = {"dpo_loss": [], "gd_loss": []}

    os.makedirs("res/dpo_gd", exist_ok=True)

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

            # === 1. DPO Loss on forget set ===
            input_ids_forget = forget_batch["input_ids"].to(device)
            dpo_loss = compute_npo_loss(model, initial_model, input_ids_forget, beta=beta)

            # === 2. CrossEntropy GD on retain set ===
            input_ids_retain = retain_batch["input_ids"].to(device)
            gd_loss = compute_ce_loss(model, input_ids_retain)
            loss_all = dpo_loss + gd_loss
            optimizer.zero_grad()
            loss_all.backward()
            optimizer.step()

            # Logging
            print(f"Step {step}: DPO Loss = {dpo_loss.item():.4f}, GD Loss = {gd_loss.item():.4f}")
            loss_history["dpo_loss"].append(dpo_loss.item())
            loss_history["gd_loss"].append(gd_loss.item())

        # Save model each epoch
        torch.save(model.state_dict(), f"res/npo_gd/model_epoch_{epoch+1}.pth")

    pd.DataFrame(loss_history).to_csv("res/npo_gd/loss.csv", index=False)
    print("DPO+GD Training complete. Results saved under res/dpo_gd/")

if __name__ == "__main__":
    dpo_gd_joint_train(
        epochs=10,
        lr=1e-4,
        batch_size=32,
        lora_r=4,
        beta=0.1,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
