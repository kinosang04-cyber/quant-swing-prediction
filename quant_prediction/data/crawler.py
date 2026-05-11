from FinMind.data import DataLoader

train_fild_name = '0050_train.xlsx'
test_fild_name = '0050_test.xlsx'
dl = DataLoader()
train_df = dl.taiwan_stock_daily(
    stock_id="0050",
    start_date="2020-01-01",
    end_date="2025-12-31")
train_df = train_df.drop(columns=["stock_id","Trading_money","Trading_turnover","spread"])
train_df = train_df.sort_values("date")
train_df.to_excel(train_fild_name, index=False)
test_df = dl.taiwan_stock_daily(
    stock_id="0050",
    start_date="2026-01-01",
    end_date="2026-5-10")
test_df = test_df.drop(columns=["stock_id","Trading_money","Trading_turnover","spread"])
test_df = test_df.sort_values("date")
test_df.to_excel(test_fild_name, index=False)