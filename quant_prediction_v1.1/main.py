import time
import gc
import torch
import config
import pandas as pd
from config import get_config, load_data
from torch.utils.data import DataLoader, TensorDataset
from data.crawler import crawler
from data.prepare_data import prepare_data
from train_model import train_model
from evaluate_model import evaluate_model
from Plot import plot_results

backtest_history = []
for idx in config.run_indices:
    test_star_dt = config.dataset['test_star'][idx]
    test_end_dt = config.dataset['test_end'][idx]
    train_end_dt = config.dataset['train_end'][idx]
    Term = get_config(test_star_dt, test_end_dt, train_end_dt)
    daily_record = {"date":test_end_dt, "Long_Term": 0, "Mid_Term": 0, "Short_Term":0}
    print(f"第{idx+1}天回測")
    for scale in Term['scales']:
        print(f"開始訓練：{scale['name']} ({scale['train_star']}~{scale['train_end']})")
        crawler(config.stock_id, scale['train_star'], scale['train_end'], scale['test_star'], scale['test_end'])
        time.sleep(2)
        prepare_data(config.stock_id, config.is_split, config.train_split, config.test_split, config.split_scale)
        x_train, y_train, _ = load_data("train","train")
        x_val, y_val, _     = load_data("train","val")
        x_test, y_test, dates_test = load_data("test","test")
        scale_probs = []
        for i in range(5):
            train_loader = DataLoader(TensorDataset(x_train, y_train),batch_size=config.batch_size,shuffle=False)
            val_loader = DataLoader(TensorDataset(x_val, y_val),batch_size=config.batch_size,shuffle=False)
            train_losses, val_losses = train_model(config.epochs, train_loader, val_loader, scale["learning_rate"], config.show_plot, config.device)
            date, prob = evaluate_model(x_test, y_test, dates_test, config.threshold, config.device)
            torch.cuda.empty_cache()
            scale_probs.append(prob)
        torch.cuda.empty_cache()
        scale_avg = sum(scale_probs) / len(scale_probs)
        print(f"預測未來 3 天為上漲趨勢 (模型信心度: {scale_avg:.2%})")
        daily_record[scale['name']] = scale_avg
    daily_record['date'] = date
    time_index = config.backtest_list[config.backtest_list['date'] == date].index
    if not time_index.empty:
        config.backtest_list.loc[
            time_index, ['Long_Term','Mid_Term','Short_Term']] = [
                daily_record["Long_Term"],
                daily_record["Mid_Term"],
                daily_record["Short_Term"]]
    backtest_history.append(daily_record)
    torch.cuda.empty_cache()
    gc.collect()
pd.DataFrame(backtest_history).to_excel("results/backtest_results.xlsx", index=False)
config.backtest_list.to_excel(f"data/back/{config.stock_id}_backlist.xlsx", index=False)
plot_results(config.stock_id, config.is_plot)