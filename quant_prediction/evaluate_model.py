import torch
import numpy as np
import pandas as pd
from model import Model
# import matplotlib.pyplot as plt

def evaluate_model(x_test, y_test, dates_test, threshold, show_prediction, device):
    model = Model().to(device)
    model.load_state_dict(torch.load("parameters/model.pth", map_location=device))
    model.eval()
    with torch.no_grad():
        logits = model(x_test).squeeze()
        probabilities = torch.sigmoid(logits)
        predictions = (probabilities > threshold).float()
    logits_np = logits.cpu().numpy()
    probs_np = probabilities.cpu().numpy()
    preds_np = predictions.cpu().numpy()
    label_np = y_test.cpu().numpy()
    df_result = pd.DataFrame({
        'date': dates_test,
        'logits': logits_np,
        'probability': probs_np,
        'prediction': preds_np,
        'label': label_np})
    tp_logic = ((df_result['prediction'] == 1) & (df_result['label'] == 1))
    tn_logic = ((df_result['prediction'] == 0) & (df_result['label'] == 0))
    fp_logic = ((df_result['prediction'] == 1) & (df_result['label'] == 0))
    fn_logic = ((df_result['prediction'] == 0) & (df_result['label'] == 1))
    df_result['TP'] = tp_logic.astype(float).where(df_result['label'].notna(), np.nan)
    df_result['TN'] = tn_logic.astype(float).where(df_result['label'].notna(), np.nan)
    df_result['FP'] = fp_logic.astype(float).where(df_result['label'].notna(), np.nan) # 最危險
    df_result['FN'] = fn_logic.astype(float).where(df_result['label'].notna(), np.nan)
    TP = df_result['TP'].sum()
    TN = df_result['TN'].sum()
    FP = df_result['FP'].sum()
    FN = df_result['FN'].sum()
    output_path = r'results/model_test_results.xlsx'
    df_result.to_excel(output_path, index=False)
    # 指標
    accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-9) # 判斷正確的比例
    precision = TP / (TP + FP + 1e-9) # 專注於減少 FP
    recall = TP / (TP + FN + 1e-9) # 專注減少 FN
    error_rate = FP / (TP + FP) if (TP + FP) > 0 else 0
    print("----統計結果----")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Precision: {precision:.2%}")
    print(f"Recall: {recall:.2%}")
    print(f"進場次數: {int(TP + FP)} 次")
    print(f"看錯比: {error_rate:.2%}")
    if show_prediction:
        time = dates_test[-1]
        last_prob = probs_np[-1]
        print("\n" + "="*50)
        if preds_np[-1] == 1:
            print(f"【看漲訊號】截止至 {time}")
            print(f"   預測未來 3 天為上漲趨勢 (模型信心度: {last_prob:.2%})")
        else:
            print(f"【觀望訊號】截止至 {time}")
            print(f"   目前信心低於門檻 (模型信心度: {last_prob:.2%})")
            print("="*50 + "\n")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    