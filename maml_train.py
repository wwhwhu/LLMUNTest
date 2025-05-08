import torch
from models.transformer_model import TransformerModel
from utils.data_loader import get_wikitext2_dataloader, get_wmdpbio_dataloader
from utils.lora import LoRALayer
import copy

def compute_loss(model, input_ids, labels):
    outputs = model(input_ids=input_ids, labels=labels)
    return outputs.loss

def apply_lora_to_gpt2(model, r=4):
    for block in model.model.transformer.h:
        fc_layer = block.mlp.c_fc
        block.mlp.c_fc = LoRALayer(fc_layer, r)
    return model

def meta_train(
    epochs=5,
    inner_lr=1e-3,
    meta_lr=1e-4,
    batch_size=32,
    lora_r=4,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    tokenizer = TransformerModel("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Load query (WikiText-2) and support (WM-DP-bio) dataloaders
    query_loader = get_wikitext2_dataloader(batch_size=batch_size, tokenizer=tokenizer)
    support_loader = get_wmdpbio_dataloader(batch_size=batch_size, tokenizer=tokenizer)

    # Initialize base model
    base_model = TransformerModel().to(device)
    meta_optimizer = torch.optim.Adam(base_model.parameters(), lr=meta_lr)
    loss_support = []
    loss_query = []

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        support_iter = iter(support_loader)
        for step, query_batch in enumerate(query_loader):
            try:
                support_batch = next(support_iter)
            except StopIteration:
                support_iter = iter(support_loader)
                support_batch = next(support_iter)

            # Clone model and apply LoRA to clone
            meta_model = copy.deepcopy(base_model)
            meta_model = apply_lora_to_gpt2(meta_model, r=lora_r).to(device)
            meta_model.train()

            # INNER LOOP: Train on support batch (simulates task-specific learner)
            support_input_ids = support_batch["input_ids"].to(device)
            support_labels = support_input_ids.clone()
            support_loss = compute_loss(meta_model, support_input_ids, support_labels)
            grads = torch.autograd.grad(-support_loss, meta_model.parameters(), create_graph=True)
            # Apply gradient update manually (first-order MAML)
            fast_weights = list(map(lambda p: p[1] - inner_lr * p[0], zip(grads, meta_model.parameters())))

            # OUTER LOOP: Evaluate on query batch using fast-updated parameters
            query_input_ids = query_batch["input_ids"].to(device)
            query_labels = query_input_ids.clone()

            # Temporarily assign fast weights
            fast_model = copy.deepcopy(meta_model)
            for (name, param), fast in zip(fast_model.named_parameters(), fast_weights):
                param.data = fast.data

            query_loss = compute_loss(fast_model, query_input_ids, query_labels)

            # META UPDATE: Update base model with meta gradient
            meta_optimizer.zero_grad()
            query_loss.backward()
            meta_optimizer.step()
            print(f"Step {step}: Support Loss = {support_loss.item():.4f}, Query Loss = {query_loss.item():.4f}")
            loss_support.append(support_loss.item())
            loss_query.append(query_loss.item())
            # 保存最新的模型
            torch.save(base_model.state_dict(), "res/maml/best_model.pth")
    # 保存loss为csv文件
    import pandas as pd
    loss_df = pd.DataFrame({"support_loss": loss_support, "query_loss": loss_query})
    loss_df.to_csv("res/maml/loss.csv", index=False)
    print("Training complete. Model saved to res/maml/best_model.pth")

if __name__ == "__main__":
    meta_train(
        epochs=10,
        inner_lr=1e-3,
        meta_lr=1e-4,
        batch_size=32,
        lora_r=4,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )