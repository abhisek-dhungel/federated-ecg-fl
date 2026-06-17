# training and evaluation helpers

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


def make_dataloader(X, y, batch_size=32, shuffle=True):
    X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # shape: (batch, 1, 360)
    y_tensor = torch.tensor(y, dtype=torch.long)
    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0

    for X_batch, y_batch in dataloader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        predictions = model(X_batch)
        loss = loss_fn(predictions, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X_batch.size(0)

    return total_loss / len(dataloader.dataset)


def evaluate(model, dataloader, device):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            predictions = model(X_batch)
            loss = loss_fn(predictions, y_batch)
            total_loss += loss.item() * X_batch.size(0)

            predicted_classes = predictions.argmax(dim=1)
            correct += (predicted_classes == y_batch).sum().item()
            total += y_batch.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def train_locally(model, X_train, y_train, device, epochs=5, lr=0.001):
    # same function used in federated client and local-only baseline
    model.to(device)
    dataloader = make_dataloader(X_train, y_train)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        train_one_epoch(model, dataloader, optimizer, device)

    return model


def split_train_test(X, y, test_ratio=0.2, seed=42):
    # split inside each patient, not across patients
    rng = np.random.RandomState(seed)
    n = len(X)
    indices = rng.permutation(n)
    test_size = int(n * test_ratio)

    test_idx = indices[:test_size]
    train_idx = indices[test_size:]

    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]
