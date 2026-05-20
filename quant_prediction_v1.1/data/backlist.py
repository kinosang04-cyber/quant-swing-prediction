import json
import numpy as np
import pandas as pd
from FinMind.data import DataLoader

def backlist_crawler(stock_id, back_star , back_end):
    file_path = f'data/back/{stock_id}_backtest_dataset.json'
    backlist_fild_name = f'data/back/{stock_id}_backlist.xlsx'
    window=30
    dl = DataLoader()
    backlist_df = dl.taiwan_stock_daily(
        stock_id=stock_id,
        start_date=back_star,
        end_date=back_end)
    backlist_df = backlist_df.drop(columns=["stock_id","Trading_money","Trading_turnover","spread"])
    backlist_df = backlist_df.sort_values("date")
    backlist_df["Long_Term"] = np.nan
    backlist_df["Mid_Term"] = np.nan
    backlist_df["Short_Term"] = np.nan
    backlist_df.excel = backlist_df.iloc[window :].copy()
    backlist_df.excel.to_excel(backlist_fild_name, index=False)
    df_time = pd.to_datetime(backlist_df['date'])
    star = []
    end = []
    for i in range(len(df_time) - window + 1):
        x_window = df_time.iloc[i:i+window]
        t_start = x_window.iloc[0]
        t_end = x_window.iloc[-1]
        star.append([t_start.year, t_start.month, t_start.day])
        end.append([t_end.year, t_end.month, t_end.day]) 
    dataset = {
        "org_star":star,
        "org_end":end,
        "test_star":star[1:],
        "test_end":end[1:],
        "train_end":end[:-1],
        }
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)
    
if __name__ == '__main__':
    stock_id = '0050'
    back_star = "2025-06-01"
    back_end = "2026-5-19"
    backlist_crawler(stock_id, back_star , back_end)
