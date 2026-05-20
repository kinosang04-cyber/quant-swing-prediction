from FinMind.data import DataLoader

def crawler(stock_id, train_star , train_end, test_star, test_end):
    train_fild_name = r'data/train/' + stock_id + '_train.xlsx'
    test_fild_name = r'data/test/' + stock_id + '_test.xlsx'
    dl = DataLoader()
    train_df = dl.taiwan_stock_daily(
        stock_id=stock_id,
        start_date=train_star,
        end_date=train_end)
    train_df = train_df.drop(columns=["stock_id","Trading_money","Trading_turnover","spread"])
    train_df = train_df.sort_values("date")
    train_df.to_excel(train_fild_name, index=False)
    test_df = dl.taiwan_stock_daily(
        stock_id=stock_id,
        start_date=test_star,
        end_date=test_end)
    test_df = test_df.drop(columns=["stock_id","Trading_money","Trading_turnover","spread"])
    test_df = test_df.sort_values("date")
    test_df.to_excel(test_fild_name, index=False)

if __name__ == '__main__':
    stock_id = '0050'
    train_star = "2015-03-31"
    train_end = "2026-03-31"
    test_star = "2025-06-01"
    test_end = "2026-5-19"
    train_fild_name = 'data/train/' + stock_id + '_train.xlsx'
    test_fild_name = 'data/train/' + stock_id + '_test.xlsx'
    crawler(stock_id, train_star , train_end, test_star, test_end)