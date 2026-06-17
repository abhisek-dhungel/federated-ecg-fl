# main script - federated training, local baseline, then compare

import torch
import numpy as np
import matplotlib.pyplot as plt
import flwr as fl
from flwr.simulation import run_simulation
from flwr.server import ServerApp, ServerAppComponents
from flwr.server.strategy import FedAvg
from flwr.client import ClientApp
from flwr.common import Context

from data_prep import build_client_dataset
from train_utils import split_train_test, train_locally, evaluate, make_dataloader
from model import ECGNet
from flower_client import ECGClient


PATIENT_IDS = ['100', '101', '103', '105', '108']  # one patient = one federated client
DATA_DIR = './mit-bih-arrhythmia-database-1.0.0'
INPUT_LENGTH = 360
NUM_ROUNDS = 10
EPOCHS_PER_ROUND = 5
NUM_WORST_TO_HIGHLIGHT = 5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


print(f"Loading patient data ({len(PATIENT_IDS)} clients: {', '.join(PATIENT_IDS)})...")

client_datasets = {}
for record_id in PATIENT_IDS:
    X, y = build_client_dataset(record_id, DATA_DIR)
    X_train, y_train, X_test, y_test = split_train_test(X, y)
    client_datasets[record_id] = (X_train, y_train, X_test, y_test)
    print(f"  Patient {record_id}: {len(X_train)} train beats, {len(X_test)} test beats")


print("\nRunning federated training...")

federated_accuracy_per_round = []
federated_per_client_accuracy = {}
federated_accuracy_history = {}


def make_client_app():
    def client_fn(context: Context):
        partition_id = int(context.node_config["partition-id"])
        record_id = PATIENT_IDS[partition_id]
        X_train, y_train, X_test, y_test = client_datasets[record_id]

        model = ECGNet(num_classes=5, input_length=INPUT_LENGTH)
        client = ECGClient(
            model, X_train, y_train, X_test, y_test, DEVICE,
            record_id=record_id, epochs_per_round=EPOCHS_PER_ROUND,
        )
        return client.to_client()

    return ClientApp(client_fn=client_fn)


def weighted_average(metrics):
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    total_examples = sum(num_examples for num_examples, _ in metrics)
    avg_accuracy = sum(accuracies) / total_examples

    federated_accuracy_per_round.append(avg_accuracy)

    for num_examples, m in metrics:
        record_id = m.get("record_id")
        if record_id is None:
            matches = [
                p for p in PATIENT_IDS
                if len(client_datasets[p][3]) == num_examples
            ]
            if len(matches) == 1:
                record_id = matches[0]
        if record_id is None:
            continue
        federated_per_client_accuracy[record_id] = m["accuracy"]
        federated_accuracy_history.setdefault(record_id, []).append(m["accuracy"])

    print(f"  Round complete - average accuracy across clients: {avg_accuracy:.4f}")
    return {"accuracy": avg_accuracy}


def make_server_app():
    def server_fn(context: Context):
        strategy = FedAvg(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=len(PATIENT_IDS),
            min_evaluate_clients=len(PATIENT_IDS),
            min_available_clients=len(PATIENT_IDS),
            evaluate_metrics_aggregation_fn=weighted_average,
        )
        config = fl.server.ServerConfig(num_rounds=NUM_ROUNDS)
        return ServerAppComponents(strategy=strategy, config=config)

    return ServerApp(server_fn=server_fn)


# reduced cpu here, otherwise flower simulation was too slow on my laptop
run_simulation(
    server_app=make_server_app(),
    client_app=make_client_app(),
    num_supernodes=len(PATIENT_IDS),
    backend_config={"client_resources": {"num_cpus": 0.5, "num_gpus": 0.0}},
)


print("\nTraining local-only baseline (no collaboration)...")

local_only_accuracies = {}

for record_id in PATIENT_IDS:
    X_train, y_train, X_test, y_test = client_datasets[record_id]

    model = ECGNet(num_classes=5, input_length=INPUT_LENGTH)
    model = train_locally(model, X_train, y_train, DEVICE, epochs=NUM_ROUNDS * 2)

    test_loader = make_dataloader(X_test, y_test, shuffle=False)
    loss, accuracy = evaluate(model, test_loader, DEVICE)

    local_only_accuracies[record_id] = accuracy
    print(f"  Patient {record_id} (local-only): accuracy = {accuracy:.4f}")


print("\nGenerating comparison plots...")

final_federated_accuracy = federated_accuracy_per_round[-1]
patient_labels = list(local_only_accuracies.keys())
local_values = [local_only_accuracies[p] for p in patient_labels]
federated_values = [federated_per_client_accuracy[p] for p in patient_labels]

# patients with lowest local accuracy - checking if federated model helps them
worst_patients = sorted(
    patient_labels, key=lambda p: local_only_accuracies[p]
)[:NUM_WORST_TO_HIGHLIGHT]

print("\nPer-patient comparison (local vs federated on same test set):")
for record_id in patient_labels:
    local_acc = local_only_accuracies[record_id]
    fed_acc = federated_per_client_accuracy[record_id]
    delta = fed_acc - local_acc
    marker = " <-- worst local" if record_id in worst_patients else ""
    print(
        f"  Patient {record_id}: local={local_acc:.4f}, "
        f"federated={fed_acc:.4f}, delta={delta:+.4f}{marker}"
    )

print(f"\nWorst local-only patients: {worst_patients}")
for record_id in worst_patients:
    local_acc = local_only_accuracies[record_id]
    fed_acc = federated_per_client_accuracy[record_id]
    if fed_acc > local_acc:
        print(
            f"  Patient {record_id}: federated beats local by "
            f"{fed_acc - local_acc:.4f} ({fed_acc:.4f} vs {local_acc:.4f})"
        )
    else:
        print(
            f"  Patient {record_id}: federated still below local by "
            f"{local_acc - fed_acc:.4f} ({fed_acc:.4f} vs {local_acc:.4f})"
        )

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

ax1.plot(range(1, len(federated_accuracy_per_round) + 1),
          federated_accuracy_per_round, marker='o', color='#2ca02c')
ax1.set_title(f'Federated Avg Accuracy ({NUM_ROUNDS} rounds)')
ax1.set_xlabel('Round')
ax1.set_ylabel('Average accuracy across all clients')
ax1.grid(True, alpha=0.3)

x = np.arange(len(patient_labels))
width = 0.35

ax2.bar(x - width/2, local_values, width, label='Local-only', color='#d62728')
ax2.bar(x + width/2, federated_values, width, label='Federated (global model)', color='#2ca02c')
ax2.set_xticks(x)
ax2.set_xticklabels([f'P{p}' for p in patient_labels])
ax2.set_ylabel('Test accuracy')
ax2.set_title('All Patients: Local vs Federated')
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

worst_local = [local_only_accuracies[p] for p in worst_patients]
worst_fed = [federated_per_client_accuracy[p] for p in worst_patients]
x_worst = np.arange(len(worst_patients))

ax3.bar(x_worst - width/2, worst_local, width, label='Local-only', color='#d62728')
ax3.bar(x_worst + width/2, worst_fed, width, label='Federated (global model)', color='#2ca02c')
ax3.set_xticks(x_worst)
ax3.set_xticklabels([f'Patient {p}' for p in worst_patients])
ax3.set_ylabel('Test accuracy')
ax3.set_title(f'Worst {NUM_WORST_TO_HIGHLIGHT} Local Patients vs Federated')
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('federated_vs_local_comparison.png', dpi=150)
print("\nPlot saved: federated_vs_local_comparison.png")

fig2, ax = plt.subplots(figsize=(8, 5))
for record_id in worst_patients:
    ax.plot(
        range(1, len(federated_accuracy_history[record_id]) + 1),
        federated_accuracy_history[record_id],
        marker='o', label=f'Patient {record_id}',
    )
    ax.axhline(
        local_only_accuracies[record_id], linestyle='--', alpha=0.5,
        label=f'Patient {record_id} local-only',
    )
ax.set_title('Federated Accuracy Over Rounds (Worst Local Patients)')
ax.set_xlabel('Round')
ax.set_ylabel('Test accuracy on patient test set')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('worst_patients_federated_progress.png', dpi=150)
print("Plot saved: worst_patients_federated_progress.png")

print(f"\nFinal federated average accuracy: {final_federated_accuracy:.4f}")
print(f"Local-only accuracies: {local_only_accuracies}")
print(f"Federated per-patient accuracies: {federated_per_client_accuracy}")
