# flower client - only model weights are shared, not raw ECG data

import flwr as fl
import torch
import numpy as np

from train_utils import train_locally, evaluate, make_dataloader


class ECGClient(fl.client.NumPyClient):

    def __init__(
        self, model, X_train, y_train, X_test, y_test, device,
        record_id=None, epochs_per_round=5,
    ):
        self.model = model
        self.X_train, self.y_train = X_train, y_train
        self.X_test, self.y_test = X_test, y_test
        self.device = device
        self.record_id = record_id
        self.epochs_per_round = epochs_per_round

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(np.copy(v)) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        # train only on this patient's data, then send updated weights to server
        self.model = train_locally(
            self.model, self.X_train, self.y_train,
            self.device, epochs=self.epochs_per_round, lr=0.001
        )

        updated_weights = self.get_parameters(config={})
        num_examples = len(self.X_train)
        return updated_weights, num_examples, {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)

        # check global model on this patient's test data (no training here)
        test_loader = make_dataloader(self.X_test, self.y_test, shuffle=False)
        loss, accuracy = evaluate(self.model, test_loader, self.device)

        metrics = {"accuracy": accuracy}
        if self.record_id is not None:
            metrics["record_id"] = self.record_id  # So we know which Patient this accuracyy belongs to
        return loss, len(self.X_test), metrics
