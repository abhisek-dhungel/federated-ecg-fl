# simple 1D CNN for ECG beat classification

import torch
import torch.nn as nn


class ECGNet(nn.Module):
    def __init__(self, num_classes=5, input_length=360):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=7, padding=3)  # 1D conv for ECG time series
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        self.relu = nn.ReLU()

        flattened_size = 32 * (input_length // 4)  # length reduced by pooling twice

        self.fc1 = nn.Linear(flattened_size, 64)
        self.fc2 = nn.Linear(64, num_classes)  # 5 output classes (AAMI)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool1(x)

        x = self.relu(self.conv2(x))
        x = self.pool2(x)

        x = x.flatten(start_dim=1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x


if __name__ == "__main__":
    model = ECGNet(num_classes=5, input_length=360)
    fake_batch = torch.randn(8, 1, 360)
    output = model(fake_batch)
    print("Output shape:", output.shape)
