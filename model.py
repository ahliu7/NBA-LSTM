import torch
import torch.nn as nn

class NBA_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(NBA_LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)  # LSTM output
        out = out[:, -1, :]    # Take the output of the last time step
        out = self.fc(out)     # Fully connected layer
        return self.sigmoid(out)
