import torch
from model import Model

def evaluate_model(x_test, y_test, dates_test, threshold, device):
    model = Model().to(device)
    model.load_state_dict(torch.load("parameters/model.pth", map_location=device))
    model.eval()
    with torch.no_grad():
        logits = model(x_test).squeeze()
        probabilities = torch.sigmoid(logits)
    probs_np = probabilities.cpu().numpy()
    time = dates_test[-1]
    last_prob = probs_np[-1]
    return time, last_prob
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    