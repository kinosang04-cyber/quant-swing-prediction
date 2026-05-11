import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=32):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: (batch, 14, 4)
        out, (h_n, c_n) = self.lstm(x)
        # h_n: (1, batch, hidden)
        last_hidden = h_n[-1]
        out = self.fc(last_hidden)
        return out
