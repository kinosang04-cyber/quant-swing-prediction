import os
import json
import torch
import datetime
import pandas as pd
from data.backlist import backlist_crawler
from dateutil.relativedelta import relativedelta
#=====================================================================#
stock_id = '0050'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# save path
back_star = "2025-06-01"
back_end = "2026-5-19"
# data clean/balance
is_split = True
split_scale = 4
train_split = datetime.date(2025, 6, 18).strftime("%Y-%m-%d")
test_split = datetime.date(2025, 6, 18).strftime("%Y-%m-%d")
# parameters
epochs = 600
threshold = 0.5
batch_size = 32
Long_lr = 1e-3
Mid_lr = 1e-3
Short_lr = 2e-3
# show
is_plot = True
show_plot = False
# Prediction
Prediction = True
#=====================================================================#
backtest_table_path = f"data/back/{stock_id}_backlist.xlsx"
backtest_list_path = f'data/back/{stock_id}_backtest_dataset.json'
if not (os.path.exists(backtest_table_path) and os.path.exists(backtest_list_path)):
    backlist_crawler(stock_id, back_star , back_end)
backtest_list = pd.read_excel(f"data/back/{stock_id}_backlist.xlsx")
# backtest list
with open(backtest_list_path, 'r', encoding='utf-8') as f:
    dataset = json.load(f)
if Prediction:
    run_indices = [len(dataset['test_star']) - 1]
else:
    run_indices = range(len(dataset['test_star']))
    print(f"需回測:{len(run_indices)}天")
# time setting
def get_config(test_star_list, test_end_list, train_end_list):
    train_end_dt = datetime.date(*train_end_list)
    test_star_dt = datetime.date(*test_star_list)
    test_end_dt = datetime.date(*test_end_list)
    config ={
        "train_end": train_end_dt.strftime("%Y-%m-%d"),
        "test_star": test_star_dt.strftime("%Y-%m-%d"),
        "test_end": test_end_dt.strftime("%Y-%m-%d"),
        "scales": [
        {
            "name": "Long_Term",
            "train_star": (train_end_dt-relativedelta(years=15)).strftime("%Y-%m-%d"),
            "train_end": train_end_dt.strftime("%Y-%m-%d"),
            "test_star": test_star_dt.strftime("%Y-%m-%d"),
            "test_end": test_end_dt.strftime("%Y-%m-%d"),
            "learning_rate": Long_lr
        },
        {
            "name": "Mid_Term",
            "train_star": (train_end_dt-relativedelta(years=10)).strftime("%Y-%m-%d"),
            "train_end": train_end_dt.strftime("%Y-%m-%d"),
            "test_star": test_star_dt.strftime("%Y-%m-%d"),
            "test_end": test_end_dt.strftime("%Y-%m-%d"),
            "learning_rate": Mid_lr
        },
        {
            "name": "Short_Term",
            "train_star": (train_end_dt-relativedelta(years=5)).strftime("%Y-%m-%d"),
            "train_end": train_end_dt.strftime("%Y-%m-%d"),
            "test_star": test_star_dt.strftime("%Y-%m-%d"),
            "test_end": test_end_dt.strftime("%Y-%m-%d"),
            "learning_rate": Short_lr
        }
        ]
    }
    return config
def load_data(stage1, stage2):
    path = f"data/{stage1}/{stock_id}_{stage2}_dataset.pt"
    data = torch.load(path, map_location=device)
    return data["x"], data["y"], data.get("idx", None)