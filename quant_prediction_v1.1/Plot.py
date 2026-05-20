import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf

def plot_results(stock_id, is_plot):
    path = f"data/back/{stock_id}_backlist.xlsx"
    save_path = f"results/{stock_id}_plot.png"
    df = pd.read_excel(path)
    df.index = pd.to_datetime(df['date'])
    plot_df = df[['open', 'max', 'min', 'close', 'Trading_Volume']].copy()
    plot_df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(22, 12), sharex=True, dpi=120)
    s = mpf.make_mpf_style(base_mpf_style='charles', gridstyle='--', gridcolor='lightgray')
    mpf.plot(plot_df, type='candle', ax=ax1, style=s, datetime_format='%Y-%m')
    x_axis = np.arange(len(plot_df))
    ax2.plot(x_axis, df.loc[plot_df.index, 'Long_Term'], label='Long_Term', color='blue')
    ax2.plot(x_axis, df.loc[plot_df.index, 'Mid_Term'], label='Mid_Term', color='orange')
    ax2.plot(x_axis, df.loc[plot_df.index, 'Short_Term'], label='Short_Term', color='green')
    ax2.set_xlim(ax1.get_xlim())
    xtick_indices = x_axis[::20]
    xtick_labels = [plot_df.index[i].strftime('%Y-%m') for i in xtick_indices]
    ax2.set_xticks(xtick_indices)
    ax2.set_xticklabels(xtick_labels, rotation=45)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax2.grid(True, linestyle='--', alpha=0.5)
    plt.subplots_adjust(hspace=0.05, left=0.05, right=0.95, top=0.95, bottom=0.1)
    if is_plot:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.show()