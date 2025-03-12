import torch
import torch.nn as nn
from config import Constants

hidden_size1 = Constants['hidden_size1']
hidden_size2 = Constants['hidden_size2']
num_layers = Constants['num_layers']
dropout = Constants['dropout']

class NBA_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size1=hidden_size1, hidden_size2=hidden_size2, num_layers=num_layers, dropout=dropout):
        super(NBA_LSTM, self).__init__()
        self.lstm1 = nn.LSTM(input_size=input_size,
            hidden_size=hidden_size1,
            batch_first=True,
            num_layers=num_layers,
            dropout=0 if num_layers == 1 else dropout
        )

        self.dropout1 = nn.Dropout(dropout)

        self.lstm2 = nn.LSTM(
            input_size=hidden_size1,
            hidden_size=hidden_size2,
            batch_first=True
        )

        self.dropout2 = nn.Dropout(dropout)

        self.fc = nn.Linear(hidden_size2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out1, _ = self.lstm1(x)
        out1 = self.dropout1(out1)

        out2, _ = self.lstm2(out1)
        out2 = self.dropout2(out2[:, -1, :])

        out = self.fc(out2)
        return self.sigmoid(out)
