import torch
import torch.nn as nn
import numpy as np


class ActionLSTM(nn.Module):
    def __init__(self, input_size=132, hidden_size=64, num_layers=2, num_classes=3):
        super(ActionLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def calculate_angle(a, b, c):
    """计算三个关键点之间的夹角 (用于 Agent 评估逻辑)"""
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle