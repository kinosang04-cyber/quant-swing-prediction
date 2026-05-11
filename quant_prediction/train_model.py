import torch
import torch.nn as nn
from tqdm import tqdm
from model import Model
import matplotlib.pyplot as plt
def train_model(epochs, train_loader, val_loader, learning_rate, show_plot, device):
    train_losses = []
    val_losses = []
    best_epoch_history = []
    model = Model().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_val_loss = float('inf')  # 記錄最小的驗證集損失
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5)
    epoch_pbar = tqdm(range(epochs), desc="Training Progress")
    for epoch in epoch_pbar:
        model.train()
        total_train_loss = 0
        
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device).float()
            pred = model(xb).view(-1)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        train_loss = total_train_loss / len(train_loader)
        train_losses.append(train_loss)
        model.eval() 
        total_val_loss = 0
        with torch.no_grad(): # 驗證時不計算梯度，節省內存
            for v_xb, v_yb in val_loader:
                v_xb = v_xb.to(device)
                v_yb = v_yb.to(device).float()
                v_pred = model(v_xb).view(-1)
                v_loss = criterion(v_pred, v_yb)
                total_val_loss += v_loss.item()
                
        val_loss = total_val_loss / len(val_loader)
        val_losses.append(val_loss)
        scheduler.step(val_loss)
        epoch_pbar.set_postfix(train_loss=f"{train_loss:.4f}", val_loss=f"{val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'parameters/model.pth')
            best_epoch_history.append(epoch)
    print(f"在 {best_epoch_history} epoch 時發現更好的模型")
    if show_plot:
        plt.figure(figsize=(10,6))
        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training vs Validation Loss")
        plt.legend()
        plt.grid(True)
        plt.show()
    return train_losses, val_losses
