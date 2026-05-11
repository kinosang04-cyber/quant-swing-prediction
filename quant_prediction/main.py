import os
import torch
from torch.utils.data import DataLoader, TensorDataset
from train_model import train_model
from evaluate_model import evaluate_model

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
epochs = 30
learning_rate = 1e-2
threshold = 0.85
show_plot = False
show_prediction = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_dataset = torch.load("data/0050_train_dataset.pt")
val_dataset = torch.load("data/0050_val_dataset.pt")
test_dataset = torch.load("data/0050_test_dataset.pt")
x_train = train_dataset["x"].to(device)
y_train = train_dataset["y"].to(device)
x_val = val_dataset["x"].to(device)
y_val = val_dataset["y"].to(device)
x_test = test_dataset["x"].to(device)
y_test = test_dataset["y"].to(device)
dates_test = test_dataset["idx"]
train_loader = DataLoader(
    TensorDataset(x_train, y_train),
    batch_size=32,
    shuffle=True
)
val_loader = DataLoader(
    TensorDataset(x_val, y_val),
    batch_size=32,
    shuffle=True
)
train_losses, val_losses = train_model(epochs, train_loader, val_loader, learning_rate, show_plot, device)
evaluate_model(x_test, y_test, dates_test, threshold, show_prediction, device)
