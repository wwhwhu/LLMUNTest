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

def npo_kl_train(
    epochs=5,
    lr=1e-4,
    batch_size=32,
    lora_r=4,
    kl_lambda=1.0,
    beta=1.0,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    tokenizer = TransformerModel("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # 加载数据集
    retain_loader = get_wikitext2_dataloader(batch_size=batch_size, tokenizer=tokenizer)
    forget_loader = get_wmdpbio_dataloader(batch_size=batch_size, tokenizer=tokenizer)

    # 初始化并克隆初始模型
    model = TransformerModel().to(device)
    model = apply_lora_to_gpt2(model, r=lora_r)
    initial_model = copy.deepcopy(model).eval()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_history = {"npo_loss": [], "kl_loss": [], "total_loss": []}

    os.makedirs("res/npo_kl", exist_ok=True)

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

            # === 1. 使用遗忘集计算 NPO 损失 ===
            input_ids_forget = forget_batch["input_ids"].to(device)
            loss_npo = compute_npo_loss(model, initial_model, input_ids_forget, beta=beta)

            # === 2. 在保留集上计算 KL 散度 ===
            input_ids_retain = retain_batch["input_ids"].to(device)
            current_logits = model(input_ids=input_ids_retain).logits.detach()
            with torch.no_grad():
                reference_logits = initial_model(input_ids=input_ids_retain).logits

            kl_loss = compute_kl_divergence(current_logits, reference_logits)

            # === 3. 综合损失 ===
            total_loss = loss_npo + kl_lambda * kl_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            print(f"Step {step}: NPO Loss = {loss_npo.item():.4f}, KL Loss = {kl_loss.item():.4f}, Total Loss = {total_loss.item():.4f}")
            loss_history["npo_loss"].append(loss_npo.item())
            loss_history["kl_loss"].append(kl_loss.item())
            loss_history["total_loss"].append(total_loss.item())

        # 保存每轮模型
        torch.save(model.state_dict(), f"res/npo_kl/model_epoch_{epoch+1}.pth")

    pd.DataFrame(loss_history).to_csv("res/npo_kl/loss.csv", index=False)
    print("NPO + KL training complete. Results saved to res/npo_kl/")

if __name__ == "__main__":
    npo_kl_train(
        epochs=10,
        lr=1e-4,
        batch_size=32,
        lora_r=4,
        kl_lambda=1.0,
        beta=1.0,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )