"""
Federated vs Local vs Centralized Forecasting
============================================
LSTM version (time-series aware) with fixed Flower API calls.

Run on one machine:

  # Terminal 1: server
  python federated_atic_flower_lstm.py --role server --rounds 50 --window 10

  # Terminals 2–4: three clients
  python federated_atic_flower_lstm.py --role client --cid 0 --window 10
  python federated_atic_flower_lstm.py --role client --cid 1 --window 10
  python federated_atic_flower_lstm.py --role client --cid 2 --window 10
"""

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
BIND_ADDR = "127.0.0.1:8080"        # change port if 8080 is busy

# ----------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------
import argparse, os, sys, warnings
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
import flwr as fl

warnings.filterwarnings("ignore", category=FutureWarning)

NUM_CLIENTS = 3
fed_losses:  List[float] = []
fed_maes:    List[float] = []
local_losses: List[float] = []
local_maes:   List[float] = []

# ----------------------------------------------------------------------
# Data helpers
# ----------------------------------------------------------------------
def load_raw(path: str = "sp500_full.csv") -> pd.DataFrame:
    """Load CSV with a `target` column plus any feature columns."""
    if not os.path.exists(path):
        sys.exit(f"Error: '{path}' not found.")
    df = pd.read_csv(path, index_col=0)
    df.columns = df.columns.str.lower()
    if "target" not in df.columns:
        sys.exit("CSV must contain a `target` column.")
    return df.astype(np.float32)


def create_sequences(
    X_2d: np.ndarray,
    y_1d: np.ndarray,
    window: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert 2-D rows into overlapping 3-D sequences for LSTM."""
    seq_X, seq_y = [], []
    for i in range(len(X_2d) - window):
        seq_X.append(X_2d[i : i + window])
        seq_y.append(y_1d[i + window])
    return np.array(seq_X), np.array(seq_y)


# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------
def build_model(window: int, n_features: int) -> keras.Model:
    model = keras.Sequential(
        [
            keras.layers.Input((window, n_features)),
            keras.layers.LSTM(64),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


# ----------------------------------------------------------------------
# Flower client
# ----------------------------------------------------------------------
class FLClient(fl.client.NumPyClient):
    def __init__(
        self,
        cid: int,
        train: Tuple[np.ndarray, np.ndarray],
        test: Tuple[np.ndarray, np.ndarray],
    ):
        self.cid = cid
        self.window = train[0].shape[1]
        self.n_feat = train[0].shape[2]
        self.model = build_model(self.window, self.n_feat)
        self.x_train, self.y_train = train
        self.x_test,  self.y_test  = test

    # Flower callbacks
    def get_parameters(self, config=None):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        self.model.fit(
            self.x_train, self.y_train, epochs=1, batch_size=32, verbose=0
        )
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, mae = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        return loss, len(self.x_test), {"mae": mae}


# ----------------------------------------------------------------------
# Flower orchestration
# ----------------------------------------------------------------------
def get_fed_eval_fn(test):
    def evaluate(round_num, parameters, config):
        window, n_feat = test[0].shape[1:]
        model = build_model(window, n_feat)
        model.set_weights(parameters)
        loss, mae = model.evaluate(test[0], test[1], verbose=0)
        fed_losses.append(loss); fed_maes.append(mae)
        print(f"[FedAvg] Round {round_num:02d} • MSE={loss:.6f} • MAE={mae:.6f}")
        return loss, {"mae": mae}
    return evaluate


def start_federated(test, rounds):
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=NUM_CLIENTS,
        min_evaluate_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        evaluate_fn=get_fed_eval_fn(test),
    )
    fl.server.start_server(
        server_address=BIND_ADDR,
        config=fl.server.ServerConfig(num_rounds=rounds),
        strategy=strategy,
    )


def start_client(cid, train, test):
    client = FLClient(cid, train, test)
    fl.client.start_numpy_client(
        server_address=BIND_ADDR,
        client=client,
    )


# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------
def plot_results(central_best, window):
    rounds = np.arange(1, len(fed_losses) + 1)
    lm, ls = np.mean(local_losses), np.std(local_losses)
    mm, ms = np.mean(local_maes),  np.std(local_maes)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), sharex=True)

    ax1.fill_between(rounds, lm - ls, lm + ls, alpha=0.3, color="gray")
    ax1.plot(rounds, fed_losses, "-o", markevery=5, linewidth=2, label="FedAvg")
    ax1.hlines(central_best[0], 1, rounds[-1], colors="C2",
               linestyles="--", label="Central")
    ax1.set_title(f"Test MSE (window={window})")
    ax1.set_xlabel("Round"); ax1.set_ylabel("MSE")
    ax1.grid(); ax1.legend()

    ax2.fill_between(rounds, mm - ms, mm + ms, alpha=0.3, color="gray")
    ax2.plot(rounds, fed_maes, "-o", markevery=5, linewidth=2, label="FedAvg")
    ax2.hlines(central_best[1], 1, rounds[-1], colors="C2",
               linestyles="--", label="Central")
    ax2.set_title(f"Test MAE (window={window})")
    ax2.set_xlabel("Round"); ax2.set_ylabel("MAE")
    ax2.grid(); ax2.legend()

    fig.tight_layout(); plt.show()


# ----------------------------------------------------------------------
# Main entry
# ----------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--role", choices=["server", "client"], required=True)
    p.add_argument("--cid", type=int, default=0)
    p.add_argument("--rounds", type=int, default=50)
    p.add_argument("--window", type=int, default=10, help="history length (days)")
    args = p.parse_args()

    # 1) Load raw CSV
    df = load_raw()
    features = df.drop(columns=["target"]).values
    targets  = df["target"].values

    # 2) Global train / test split
    split = int(0.8 * len(df))
    feat_train, feat_test = features[:split], features[split:]
    targ_train, targ_test = targets[:split],  targets[split:]

    # 3) Scale per feature, then create sequences
    scaler = StandardScaler().fit(feat_train)
    feat_train = scaler.transform(feat_train)
    feat_test  = scaler.transform(feat_test)

    X_train_all, y_train_all = create_sequences(feat_train, targ_train, args.window)
    X_test,      y_test      = create_sequences(feat_test,  targ_test,  args.window)

    # 4) Slice train set for clients
    size = len(X_train_all) // NUM_CLIENTS
    client_data = []
    for cid in range(NUM_CLIENTS):
        start = cid * size
        end   = (cid + 1) * size if cid < NUM_CLIENTS - 1 else len(X_train_all)
        client_data.append((X_train_all[start:end], y_train_all[start:end]))

    if args.role == "server":
        # ---- Local-only baselines ----
        for cid, (Xi, yi) in enumerate(client_data):
            m = build_model(args.window, Xi.shape[2])
            m.fit(Xi, yi, epochs=args.rounds, batch_size=32, verbose=0)
            loss, mae = m.evaluate(X_test, y_test, verbose=0)
            local_losses.append(loss); local_maes.append(mae)
            print(f"[Local] Client {cid}: MSE={loss:.6f} MAE={mae:.6f}")

        # ---- Centralized baseline ----
        best_loss = best_mae = float("inf")
        m_c = build_model(args.window, X_train_all.shape[2])
        for e in range(1, args.rounds + 1):
            m_c.fit(X_train_all, y_train_all, epochs=1, batch_size=32, verbose=0)
            loss, mae = m_c.evaluate(X_test, y_test, verbose=0)
            print(f"[Central][Epoch {e:02d}] MSE={loss:.6f} MAE={mae:.6f}")
            if loss < best_loss:
                best_loss, best_mae = loss, mae
        print(f"[Central] Best MSE={best_loss:.6f} MAE={best_mae:.6f}")

        # ---- Federated Training ----
        print("\nStarting Federated Learning …")
        start_federated((X_test, y_test), args.rounds)

        # ---- Plot ----
        plot_results((best_loss, best_mae), args.window)

    else:  # client
        if not (0 <= args.cid < NUM_CLIENTS):
            sys.exit("Error: cid out of range.")
        start_client(args.cid, client_data[args.cid], (X_test, y_test))


if __name__ == "__main__":
    main()
