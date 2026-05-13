import numpy as np
import pandas as pd

strategy_rules = """
回測規則:
1. 進場條件：模型預測機率 p >= 0.7，以當日收盤價 (Close) 買入。
2. 風控邏輯 (T+1 ~ T+2)：
   - 停損：若最低價 (Min) 低於買價 1%，以最低價出場。
   - 停利：若收盤價 (Close) 高於買價 3%，以收盤價出場。
3. 強制出場：若前兩日未觸發條件，於 T+3 日以收盤價 (Close) 出場。
4. 摩擦成本：考慮 0.1% (0.001) 交易手續費與稅。
"""
result_path = "results/model_test_results.xlsx"
real_path = "data/0050_test.xlsx"
result = pd.read_excel(result_path)
real_data = pd.read_excel(real_path)
vol_columns = ['open', 'max', 'min', 'close']
buy_points = []
sell_points = []
date = result['date']
for d in date:
    p = result[result['date'] == d]['probability'].item()
    if (p >= 0.7):
        idx = real_data.index[real_data['date'] == d].item()
        if idx + 3 >= len(real_data):
            continue
        future_1day = real_data.at[idx + 1, 'close']
        future_2day = real_data.at[idx + 2, 'close']
        future_3day = real_data.at[idx + 3, 'close']
        deadline_1day = real_data.at[idx + 1, 'min']
        deadline_2day = real_data.at[idx + 2, 'min']
        buy = real_data[real_data['date'] == d]['close'].item()
        buy_points.append(buy)
        if deadline_1day < buy*0.99:
            sell_points.append(deadline_1day)
            continue
        if deadline_2day < buy*0.99:
            sell_points.append(deadline_2day)
            continue
        if future_1day > buy*1.03:
            sell_points.append(future_1day)
            continue
        if future_2day > buy*1.03:
            sell_points.append(future_2day)
            continue
        sell_points.append(future_3day)
sell_points = np.array(sell_points)
buy_points = np.array(buy_points)
raw_profit = (sell_points - buy_points).sum() * 1000
buy_cost = buy_points.sum() * 1000
fee_rate = 0.001  
net_profit = raw_profit - (buy_cost * fee_rate) 
roi = (net_profit / buy_cost) * 100
win_rate = (sell_points > buy_points).sum() / len(buy_points)
print(strategy_rules)
print(f"總買進次數: {len(buy_points)}")
print(f"總損益: {raw_profit.astype(int)}")
print(f"投入總成本: {buy_cost.astype(int)}")
print(f"平均投入成本: {(buy_cost/3).astype(int)}")
print(f"考慮交易摩擦成本損益: {net_profit.astype(int)}")
print(f"獲利比: {roi:.2f}%")
print(f"策略勝率: {win_rate:.2%}")










