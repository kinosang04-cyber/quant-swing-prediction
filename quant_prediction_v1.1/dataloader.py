import torch
import config

def load_data(stage1, stage2):
    path = f"data/{stage1}/{config.stock_id}_{stage2}_dataset.pt"
    data = torch.load(path, map_location=config.device)
    return data["x"], data["y"], data.get("idx", None)