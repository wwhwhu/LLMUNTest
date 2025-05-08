import torch
import torch.nn.functional as F
from models.transformer_model import TransformerModel
from utils.data_loader import get_wikitext2_dataloader, get_wmdpbio_dataloader
from utils.lora import LoRALayer
import pandas as pd
import os

def compute_loss(model, input_ids, labels):
    outputs = model(input_ids=input_ids, labels=labels)
    return outputs.loss

def get_logits(model, input_ids):
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        return outputs.logits

def compute_kl_divergence(current_logits, reference_logits):
    current_log_probs = F.log_softmax(current_logits, dim=-1)
    reference_probs = F.softmax(reference_logits, dim=-1)
    kl = F.kl_div(current_log_probs, reference_probs, reduction='batchmean')
    return kl

def apply_lora_to_gpt2(model, r=4):
    for block in model.model.transformer.h:
        fc_layer = block.mlp.c_fc
        block.mlp.c_fc = LoRALayer(fc_layer, r)
    return model

def ga_kl_train(
    epochs=5,
    lr=1e-4,
    batch_size=32,
    lora_r=4,
    kl_lambda=1.0,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    tokenizer = TransformerModel("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Load dataloaders
    retain_loader = get_wikitext2_dataloader(batch_size=batch_size, tokenizer=tokenizer)
    forget_loader = get_wmdpbio_dataloader(batch_size=batch_size, tokenizer=tokenizer)

    # Initialize and clone the base model
    model = TransformerModel().to(device)
    model = apply_lora_to_gpt2(model, r=lora_r)
    initial_model = copy.deepcopy(model).eval()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_history = {"ga_loss": [], "kl_loss": [], "total_loss": []}

    os.makedirs("res/ga_kl", exist_ok=True)

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

            # === 1. Compute GA loss ===
            input_ids_forget = forget_batch["input_ids"].to(device)
            labels_forget = input_ids_forget.clone()
            loss_forget = compute_loss(model, input_ids_forget, labels_forget)

            # === 2. Compute KL divergence on retain set ===
            input_ids_retain = retain_batch["input_ids"].to(device)
            current_logits = model(input_ids=input_ids_retain).logits.detach()
            with torch.no_grad():
                reference_logits = initial_model(input_ids=input_ids_retain).logits

            kl_loss = compute_kl_divergence(current_logits, reference_logits)

            # === 3. Combined loss ===
            total_loss = -loss_forget + kl_lambda * kl_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            print(f"Step {step}: GA Loss = {loss_forget.item():.4f}, KL Loss = {kl_loss.item():.4f}, Total Loss = {total_loss.item():.4f}")
            loss_history["ga_loss"].append(loss_forget.item())
            loss_history["kl_loss"].append(kl_loss.item())
            loss_history["total_loss"].append(total_loss.item())

        # Save model
        torch.save(model.state_dict(), f"res/ga_kl/model_epoch_{epoch+1}.pth")

    pd.DataFrame(loss_history).to_csv("res/ga_kl/loss.csv", index=False)
    print("GA + KL training complete. Results saved to res/ga_kl/")

if __name__ == "__main__":
    import copy
    ga_kl_train(
        epochs=10,
        lr=1e-4,
        batch_size=32,
        lora_r=4,
        kl_lambda=1.0,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
