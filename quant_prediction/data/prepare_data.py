import torch
import pandas as pd
from functions import division, split, features

feature_col = ['vol_ratio', 'k_strength', 'z_score', 'red_ratio']  
window = 14
train_fild = '0050_train.xlsx'
test_fild = '0050_test.xlsx'
train_path = '0050_train_dataset.pt'
val_path = '0050_val_dataset.pt'
test_path = '0050_test_dataset.pt'
train_df = pd.read_excel(train_fild)
train_df = split(train_df, "2025-06-18", 4)
train_df = features(train_df)
train_df = train_df.dropna()
train_df = division(train_df, window, feature_col)
N = len(train_df["x"])
splits = int(N * 0.8)
train_dataset = {
    "x": torch.tensor(train_df['x'][:splits], dtype=torch.float32),
    "y": torch.tensor(train_df['y'][:splits], dtype=torch.float32),
    "idx": train_df['idx'][:splits],
    "window": 14
    }
val_dataset = {
    "x": torch.tensor(train_df['x'][splits:], dtype=torch.float32),
    "y": torch.tensor(train_df['y'][splits:], dtype=torch.float32),
    "idx": train_df['idx'][splits:],
    "window": 14
    }
test_df = pd.read_excel(test_fild)
test_df = features(test_df)
test_df = division(test_df, window, feature_col)
test_dataset = {
    "x": torch.tensor(test_df['x'], dtype=torch.float32),
    "y": torch.tensor(test_df['y'], dtype=torch.float32),
    "idx": test_df['idx'],
    "window": 14}
torch.save(train_dataset, train_path)
torch.save(val_dataset, val_path)
torch.save(test_dataset, test_path)