import torch
from models.transformer_model import TransformerModel
from utils.data_loader import get_wmdpbio_dataloader
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

def ga_train_on_forget_set(
    epochs=5,
    lr=1e-3,
    batch_size=32,
    lora_r=4,
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

    loss_history = []

    os.makedirs("res/ga", exist_ok=True)

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        model.train()
        for step, batch in enumerate(support_loader):
            input_ids = batch["input_ids"].to(device)
            labels = input_ids.clone()

            loss = compute_loss(model, input_ids, labels)

            # Gradient Ascent: maximize loss => minimize (-loss)
            optimizer.zero_grad()
            (-loss).backward()
            optimizer.step()

            print(f"Step {step}: GA Loss = {loss.item():.4f}")
            loss_history.append(loss.item())

        # 保存每轮的模型参数
        torch.save(model.state_dict(), f"res/ga/model_epoch_{epoch+1}.pth")

    # 保存loss曲线
    loss_df = pd.DataFrame({"ga_loss": loss_history})
    loss_df.to_csv("res/ga/loss.csv", index=False)
    print("GA Training complete. Models and losses saved under res/ga/")

if __name__ == "__main__":
    ga_train_on_forget_set(
        epochs=10,
        lr=1e-3,
        batch_size=32,
        lora_r=4,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
