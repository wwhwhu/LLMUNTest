import torch
from models.transformer_model import TransformerModel
from utils.data_loader import get_wikitext2_dataloader, get_wmdpbio_dataloader
from utils.lora import LoRALayer
import pandas as pd
import os

def compute_activation_loss(model, input_ids, reference_activations=None, c=10.0, alpha=1.0, u=None):
    model_outputs = model(input_ids, output_hidden_states=True)
    hidden_states = model_outputs.hidden_states[-1]  # Last layer activation: [B, L, H]

    if reference_activations is None:
        # Forget loss: drive activations in random direction
        if u is None:
            u = torch.rand(hidden_states.size(-1), device=hidden_states.device)  # [H]
            u = u / u.norm()  # Normalize
        target = c * u  # Scale random unit vector
        loss = ((hidden_states - target) ** 2).mean()
    else:
        # Retain loss: preserve original activations
        loss = ((hidden_states - reference_activations) ** 2).mean()

    return loss

def apply_lora_to_gpt2(model, r=4):
    for block in model.model.transformer.h:
        fc_layer = block.mlp.c_fc
        block.mlp.c_fc = LoRALayer(fc_layer, r)
    return model

def rmu_train(
    epochs=5,
    lr=1e-4,
    batch_size=32,
    lora_r=4,
    alpha=1.0,
    c=10.0,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    tokenizer = TransformerModel("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Load dataloaders
    retain_loader = get_wikitext2_dataloader(batch_size=batch_size, tokenizer=tokenizer)
    forget_loader = get_wmdpbio_dataloader(batch_size=batch_size, tokenizer=tokenizer)

    # Initialize model
    model = TransformerModel().to(device)
    model = apply_lora_to_gpt2(model, r=lora_r)

    # Clone frozen model for retain guidance
    frozen_model = TransformerModel().to(device)
    frozen_model.load_state_dict(model.state_dict())
    frozen_model.eval()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_history = {"forget_loss": [], "retain_loss": []}

    os.makedirs("res/rmu", exist_ok=True)

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

            input_ids_forget = forget_batch["input_ids"].to(device)
            input_ids_retain = retain_batch["input_ids"].to(device)

            with torch.no_grad():
                retain_ref_activations = frozen_model(input_ids_retain, output_hidden_states=True).hidden_states[-1]

            # Forget loss
            u = torch.rand(retain_ref_activations.size(-1), device=device)
            u = u / u.norm()
            loss_forget = compute_activation_loss(model, input_ids_forget, reference_activations=None, c=c, u=u)

            # Retain loss
            loss_retain = compute_activation_loss(model, input_ids_retain, reference_activations=retain_ref_activations)

            # Total loss
            total_loss = loss_forget + alpha * loss_retain
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            print(f"Step {step}: Forget Loss = {loss_forget.item():.4f}, Retain Loss = {loss_retain.item():.4f}")
            loss_history["forget_loss"].append(loss_forget.item())
            loss_history["retain_loss"].append(loss_retain.item())

        # 保存模型
        torch.save(model.state_dict(), f"res/rmu/model_epoch_{epoch+1}.pth")

    pd.DataFrame(loss_history).to_csv("res/rmu/loss.csv", index=False)
    print("RMU Training complete. Results saved under res/rmu/")

if __name__ == "__main__":
    rmu_train(
        epochs=10,
        lr=1e-4,
        batch_size=32,
        lora_r=4,
        alpha=1.0,
        c=10.0,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
