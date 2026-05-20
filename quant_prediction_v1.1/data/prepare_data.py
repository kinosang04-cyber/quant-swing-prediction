import torch
import pandas as pd
import numpy as np
from .functions import division, split, features

def prepare_data(stock_id, is_split, train_split, test_split, split_scale):
    feature_col = ['vol_ratio', 'k_strength', 'z_score', 'red_ratio']  
    window = 14
    train_fild = 'data/train/' + stock_id + '_train.xlsx'
    test_fild = 'data/test/' + stock_id + '_test.xlsx'
    train_path = 'data/train/' + stock_id + '_train_dataset.pt'
    val_path = 'data/train/' + stock_id + '_val_dataset.pt'
    test_path = 'data/test/' + stock_id + '_test_dataset.pt'
    train_df = pd.read_excel(train_fild)
    if is_split:
        train_df = split(train_df, train_split, split_scale)
    train_df = features(train_df)
    train_df = train_df.dropna()
    train_df = division(train_df, window, feature_col)
    N = len(train_df["x"])
    splits = int(N * 0.8)
    x_data_t = np.array(train_df['x'][:splits])
    y_data_t = np.array(train_df['y'][:splits])
    x_data_v = np.array(train_df['x'][splits:])
    y_data_v = np.array(train_df['y'][splits:])
    train_dataset = {
        "x": torch.tensor(x_data_t, dtype=torch.float32),
        "y": torch.tensor(y_data_t, dtype=torch.float32),
        "idx": train_df['idx'][:splits],
        "window": 14
        }
    val_dataset = {
        "x": torch.tensor(x_data_v, dtype=torch.float32),
        "y": torch.tensor(y_data_v, dtype=torch.float32),
        "idx": train_df['idx'][splits:],
        "window": 14
        }
    test_df = pd.read_excel(test_fild)
    if is_split:
        test_df = split(test_df, test_split, split_scale)
    test_df = features(test_df)
    test_df = division(test_df, window, feature_col)
    test_dataset = {
        "x": torch.tensor(np.array(test_df['x']), dtype=torch.float32),
        "y": torch.tensor(np.array(test_df['y']), dtype=torch.float32),
        "idx": test_df['idx'],
        "window": 14}
    torch.save(train_dataset, train_path)
    torch.save(val_dataset, val_path)
    torch.save(test_dataset, test_path)
    