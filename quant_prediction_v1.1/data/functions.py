import numpy as np

def division(df, window=14, feature_col=['vol_ratio', 'k_strength', 'z_score', 'red_ratio']):
    x = []
    y = []
    idx = []
    for i in range(len(df) - window + 1):
        x_window = df.iloc[i:i+window][feature_col].values
        label = df.iloc[i+window-1]['label']
        x.append(x_window)
        y.append(label)
        current_date = df.iloc[i + window - 1]['date']
        idx.append(current_date)   
    dataset = {
        "x":x,
        "y":y,
        "idx":idx,
        "window":14,
        }
    return dataset

def split(df, split_date, scale):
    if split_date in df["date"].values:
        split_idx = df[df['date'] == split_date].index[0]
        col_to_fix = ['open','close','max','min']
        df.loc[:split_idx, col_to_fix] = df.loc[:split_idx, col_to_fix] / scale
        df.loc[:split_idx, 'Trading_Volume'] = df.loc[:split_idx, 'Trading_Volume'] * scale
        print(f"已根據日期 {split_date} 完成調整")
    return df

def features(df):
    ma14 = df['close'].shift(1).rolling(window=14, min_periods=1).mean()
    std14 = df['close'].shift(1).rolling(window=14, min_periods=1).std()
    future_mean = (
        df['close'].shift(-1) +
        df['close'].shift(-2) +
        df['close'].shift(-3)) / 3
    future_3day = df['close'].shift(-3)
    df['label'] = ((future_mean > ma14*1.01) & (future_3day > ma14*1.006)).astype(int)
    df['label'] = df['label'].where(df.index[-3] > df.index, np.nan)
    df.iloc[-3:, df.columns.get_loc('label')] = np.nan
    df['vol_ratio'] = df['Trading_Volume'] / df['Trading_Volume'].shift(1).rolling(window=14, min_periods=1).mean()
    df['k_strength'] = abs(df['close'] - df['open']) / (df['max'] - df['min'] + 1e-6)
    df['z_score'] = abs(df['close'] - ma14) / (std14 + 1e-6)
    df['color'] = ((df['close'] - df['open']) > 0).astype(int)
    df['red_ratio'] = df['color'].shift(1).rolling(window=14, min_periods=0).mean()
    df.drop(columns=['color'])
    df = df.iloc[14:].reset_index(drop=True)
    return df