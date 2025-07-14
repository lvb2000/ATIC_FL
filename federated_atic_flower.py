"""
Federated vs Local vs Centralized Comparison for ATIC S&P-500 Forecasting
========================================================================

This script compares three modes of training:

 1. **Local-only**: each client trains on its slice for `rounds` epochs.
 2. **Centralized**: a single model trained on the full training set, tracking best epoch.
 3. **Federated (FedAvg)**: three-client federated learning, 1 epoch per round, requiring all clients.

It then produces a publication-ready two-panel figure showing:
 - Test MSE over federated rounds
 - Test MAE over federated rounds

With:
 - Gray bands for the mean ±1σ of local-only (clients trained in isolation)
 - Solid line for FedAvg trajectory
 - Dashed line for centralized best result

Usage:
  # Server (computes all and shows plots):
  python federated_atic_flower.py --role server --rounds 50

  # Clients (in three separate terminals):
  python federated_atic_flower.py --role client --cid 0
  python federated_atic_flower.py --role client --cid 1
  python federated_atic_flower.py --role client --cid 2

Prerequisites:
  - `sp500_full.csv` in the same folder, with a 'target' column.
  - Install dependencies:
      pip install flwr[simulation] pandas scikit-learn tensorflow matplotlib
"""

import argparse
import sys
import os
import warnings
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
import flwr as fl
import random
import tensorflow as tf

warnings.filterwarnings("ignore", category=FutureWarning)

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

parser = argparse.ArgumentParser()
parser.add_argument("--role",   choices=["server", "client"], required=True)
parser.add_argument("--cid",    type=int, default=0)
parser.add_argument("--rounds", type=int, default=50)
parser.add_argument("--data_dist", default="random",choices=["random", "quarters"], type=str)
parser.add_argument("--num_clients", default= 3, type=int)
parser.add_argument("--compare_dataset", default=False, type = bool)
parser.add_argument("--epoch_variance", default=False, type = bool)
parser.add_argument("--smoothing", action="store_false", dest="smoothing", help="Disable smoothing (default: enabled)")
parser.add_argument("--retrain", default=False, type = bool)
parser.set_defaults(smoothing=True)
args = parser.parse_args()

NUM_CLIENTS = args.num_clients
fed_losses: List[float] = []
fed_maes:   List[float] = []

local_hist_losses = np.zeros((NUM_CLIENTS, 0)).tolist()   # will become lists of length = rounds
local_hist_maes   = np.zeros((NUM_CLIENTS, 0)).tolist()

central_hist_loss = []
central_hist_mae  = []

def load_features(path: str = "sp500_full.csv") -> pd.DataFrame:
    if not os.path.exists(path):
        sys.exit(f"Error: '{path}' not found.")
    df = pd.read_csv(path, index_col=0)
    df.columns = df.columns.str.lower()
    return df.astype(np.float32)


def build_model(input_dim: int) -> keras.Model:
    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(150, activation="relu"),
        keras.layers.Dense(10, activation="relu"),
        keras.layers.Dense(1, activation="linear"),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


class FLClient(fl.client.NumPyClient):
    def __init__(
        self,
        cid: int,
        train_data: Tuple[np.ndarray, np.ndarray],
        test_data:  Tuple[np.ndarray, np.ndarray]
    ):
        self.cid = cid
        self.model = build_model(train_data[0].shape[1])
        self.x_train, self.y_train = train_data
        self.x_test, self.y_test   = test_data

    def get_parameters(self, config=None):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        base_epochs = config.get("base_epochs", 1)
        round_num = int(config.get("round_num", 1))
        # Set base_epochs to 0 if round_num < cid*10, else use the provided base_epochs
        if args.epoch_variance:
            if (round_num-1) % (self.cid + 1) != 0:
                base_epochs = 0
        self.model.fit(
            self.x_train,
            self.y_train,
            epochs=base_epochs,
            batch_size=32,
            verbose=0,
        )
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, mae = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        return loss, len(self.x_test), {"mae": mae}
    
_best_mae, _best_mae_weights = float("inf"), None
def get_fed_eval_fn(test_data: Tuple[np.ndarray, np.ndarray]):
    def evaluate(round_num: int, parameters, config):
        model = build_model(test_data[0].shape[1])
        model.set_weights(parameters)
        loss, mae = model.evaluate(test_data[0], test_data[1], verbose=0)
        global _best_mae, _best_mae_weights
        if mae < _best_mae:
            _best_mae = mae
            _best_mae_weights = parameters
        # ----- NEW: skip the initial round-0 measurement -----
        if round_num == 0:          # baseline, keep lists at 50 points
            return loss, {"mae": mae}

        fed_losses.append(loss)
        fed_maes.append(mae)
        print(f"[FedAvg] Round {round_num:02d}  •  MSE={loss:.6f}  •  MAE={mae:.6f}")
        return loss, {"mae": mae}
    return evaluate

def start_federated(test_data: Tuple[np.ndarray, np.ndarray], num_rounds: int) -> None:
    def fit_config_fn(server_round: int):
        return {"round_num": str(server_round),"base_epochs": 1}
    # Require all clients each round
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=NUM_CLIENTS,
        min_evaluate_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        evaluate_fn=get_fed_eval_fn(test_data),
        on_fit_config_fn=fit_config_fn,
    )
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )
    


def start_client(
    cid: int,
    train_data: Tuple[np.ndarray, np.ndarray],
    test_data:  Tuple[np.ndarray, np.ndarray]
) -> None:
    client = FLClient(cid, train_data, test_data)
    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=client,
    )

def smooth_results(local_hist_losses, local_hist_maes,
                   fed_losses, fed_maes,
                   central_hist_loss, central_hist_mae,
                   window=5,
                   retrain_hist_loss=None, retrain_hist_mae=None,
                   scratch_hist_loss=None,scratch_hist_mae=None):

    def running_average(x, w):
        x = np.asarray(x)
        if len(x) < w:
            return x
        return np.convolve(x, np.ones(w)/w, mode='valid')

    # Smooth local_hist_losses and local_hist_maes per client
    smoothed_local_hist_losses = [running_average(losses, window).tolist() for losses in local_hist_losses]
    smoothed_local_hist_maes   = [running_average(maes, window).tolist() for maes in local_hist_maes]

    # Smooth the rest (1D lists)
    smoothed_fed_losses        = running_average(fed_losses, window).tolist()
    smoothed_fed_maes          = running_average(fed_maes, window).tolist()
    smoothed_central_hist_loss = running_average(central_hist_loss, window).tolist()
    smoothed_central_hist_mae  = running_average(central_hist_mae, window).tolist()

    # Optionally smooth retrain results
    if retrain_hist_loss is not None and retrain_hist_mae is not None:
        smoothed_retrain_hist_loss = running_average(retrain_hist_loss, window).tolist()
        smoothed_retrain_hist_mae  = running_average(retrain_hist_mae, window).tolist()
    else:
        smoothed_retrain_hist_loss = None
        smoothed_retrain_hist_mae  = None

    # Optionally smooth retrain results
    if scratch_hist_loss is not None and scratch_hist_mae is not None:
        smoothed_scratch_hist_loss = running_average(scratch_hist_loss, window).tolist()
        smoothed_scratch_hist_mae  = running_average(scratch_hist_mae, window).tolist()
    else:
        smoothed_scratch_hist_loss = None
        smoothed_scratch_hist_mae  = None

    return (smoothed_local_hist_losses, smoothed_local_hist_maes,
            smoothed_fed_losses, smoothed_fed_maes,
            smoothed_central_hist_loss, smoothed_central_hist_mae,
            smoothed_retrain_hist_loss, smoothed_retrain_hist_mae,
            smoothed_scratch_hist_loss, smoothed_scratch_hist_mae)

def plot_paper_figure(local_hist_losses, local_hist_maes,
                      fed_losses, fed_maes,
                      central_hist_loss, central_hist_mae,
                      retrain_hist_loss=None, retrain_hist_mae=None,
                      scratch_hist_loss=None,scratch_hist_mae=None):
    
    # Apply smoothing if enabled
    if args.smoothing:
        (local_hist_losses, local_hist_maes,
         fed_losses, fed_maes,
         central_hist_loss, central_hist_mae,
         retrain_hist_loss, retrain_hist_mae,
         scratch_hist_loss,scratch_hist_mae) = smooth_results(
            local_hist_losses, local_hist_maes,
            fed_losses, fed_maes,
            central_hist_loss, central_hist_mae,
            window=3,
            retrain_hist_loss=retrain_hist_loss,
            retrain_hist_mae=retrain_hist_mae,
            scratch_hist_loss=scratch_hist_loss,
            scratch_hist_mae=scratch_hist_mae
        )
    
    rounds = np.arange(1, len(local_hist_losses[0]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), sharex=True)

    # --- MSE panel -------------------------------------------------
    # Assign each local a unique color, FedAvg a distinct color (e.g., C0), and centralized as black dashed
    for cid, losses in enumerate(local_hist_losses):
        ax1.plot(rounds, losses, color=f"C{cid+1}", linewidth=1,
                 label=f"Local {cid}", alpha=0.7)
    ax1.plot(rounds, central_hist_loss, "k--", linewidth=2,
             label="Centralised")
    ax1.plot(rounds, fed_losses, color="C0", linewidth=2,
             label="FedAvg")
    # Add retrain curve if present
    if retrain_hist_loss is not None:
        rounds_retrain = np.arange(0, len(retrain_hist_loss))
        ax1.plot(rounds_retrain, retrain_hist_loss, color="C3", linewidth=2, linestyle=":", label="FedAvg retrain last-layer")
        # Add retrain curve if present
    if scratch_hist_loss is not None:
        rounds_retrain = np.arange(0, len(scratch_hist_loss))
        ax1.plot(rounds_retrain, scratch_hist_loss, color="C4", linewidth=2, linestyle=":", label="FedAvg retrain from scratch")
                 
    # Load dataset using data_loading()
    data = data_loading()
    y_test = data['y_test']

    # Predict zeros
    y_pred_zeros = np.zeros_like(y_test)

    # Compute MSE loss (mean squared error)
    mse_loss = np.mean((y_test - y_pred_zeros) ** 2)
    # Compute MAE (mean absolute error) for zero predictor
    mae_loss = np.mean(np.abs(y_test - y_pred_zeros))

    ax1.axhline(mse_loss, color="gray", linestyle="--", linewidth=1.5, label="Zero predictor")

    ax1.set_title("Test MSE")
    ax1.set_xlabel("Epoch / Round")
    ax1.set_ylabel("MSE")
    ax1.grid(alpha=0.3)

    # --- MAE panel -------------------------------------------------
    for cid, maes in enumerate(local_hist_maes):
        # Do not add labels again for the same lines
        ax2.plot(rounds, maes, color=f"C{cid+1}", linewidth=1, alpha=0.7)
    ax2.plot(rounds, central_hist_mae, "k--", linewidth=2)
    ax2.plot(rounds, fed_maes, "C0", linewidth=2)
    if retrain_hist_mae is not None:
        rounds_retrain = np.arange(0, len(retrain_hist_mae))
        ax2.plot(rounds_retrain, retrain_hist_mae, color="C3", linewidth=2, linestyle=":")
    if retrain_hist_mae is not None:
        rounds_retrain = np.arange(0, len(scratch_hist_mae))
        ax2.plot(rounds_retrain, scratch_hist_mae, color="C4", linewidth=2, linestyle=":")

    # Print MAE loss for zero predictor as a horizontal line
    ax2.axhline(mae_loss, color="gray", linestyle="--", linewidth=1.5, label="Zero predictor")

    ax2.set_title("Test MAE")
    ax2.set_xlabel("Epoch / Round")
    ax2.set_ylabel("MAE")
    ax2.grid(alpha=0.3)

    # Put a single legend centered below both axes, only using handles/labels from ax1
    handles, labels = ax1.get_legend_handles_labels()
    # Move the legend inside the figure area, below the plots but above the x-axis labels
    fig.legend(handles, labels, loc="lower center", ncol=5)
    fig.tight_layout(rect=[0, 0.12, 1, 1])
    plt.show()

def random_client_data(n_samples, X_train_all, y_train_all):
    # Client slices
    client_slices = []
    all_indeces = np.random.permutation(np.arange(n_samples))
    size = n_samples // NUM_CLIENTS
    for cid in range(NUM_CLIENTS):
        start = cid * size
        stop = (cid + 1) * size if cid < NUM_CLIENTS - 1 else n_samples
        client_indeces = all_indeces[start:stop]
        client_slices.append((X_train_all[client_indeces], y_train_all[client_indeces]))
    return client_slices

def quarters_client_data(months,X_train_all, y_train_all):
    """
    Distribute data to clients by assigning each client a part of the year (months).
    For 4 clients, each gets 3 months (quarters). For more clients, months are split as evenly as possible.
    """
    client_slices = []
    # Always 4 clients, each gets 3 months (quarters)
    month_ranges = [(1, 3), (4, 6), (7, 9), (10, 12)]
    for m_start, m_end in month_ranges:
        client_indices = np.where((months >= m_start) & (months <= m_end))[0]
        client_slices.append((X_train_all[client_indices], y_train_all[client_indices]))
    return client_slices

def retrain_data_loading(data,scaler):
    df = load_features("FTSE100_processed.csv")
    split = int(len(df) * 0.8)
    train_df, test_df = df.iloc[:split], df.iloc[split:]
    X_test = test_df.drop(columns=["target"]).to_numpy()
    y_test = test_df["target"].to_numpy()
    X_train_scaled = scaler.transform(train_df.drop(columns=["target"]).to_numpy())
    y_train = train_df["target"].to_numpy()
    X_test_scaled = scaler.transform(X_test)
    data["X_train_retrain"] = X_train_scaled
    data["y_train_retrain"] = y_train
    data["X_test_retrain"] = X_test_scaled
    data["y_test_retrain"] = y_test
    return data

def data_loading():
    # Load and split
    df = load_features()
    split = int(len(df) * 0.8)
    train_df, test_df = df.iloc[:split], df.iloc[split:]
    # Prepare test data
    X_test = test_df.drop(columns=["target"]).to_numpy()
    y_test = test_df["target"].to_numpy()
    # Global scaling
    scaler = StandardScaler().fit(train_df.drop(columns=["target"]).to_numpy())
    X_train_all = scaler.transform(train_df.drop(columns=["target"]).to_numpy())
    y_train_all = train_df["target"].to_numpy()
    X_test_scaled = scaler.transform(X_test)
    # Compute client slices
    if args.data_dist == 'random':
        client_slices = random_client_data(len(train_df), X_train_all, y_train_all)
    elif args.data_dist == 'quarters':
        if args.num_clients == 4:
            client_slices = quarters_client_data(train_df['month'].to_numpy(), X_train_all, y_train_all)
        else:
            sys.exit("Error: quarters data_dist requires num_clients == 4")

    # Store all data in a dict
    data_dict = {
        "X_train_all": X_train_all,
        "y_train_all": y_train_all,
        "X_test": X_test_scaled,
        "y_test": y_test,
        "client_slices": client_slices,
    }
    if args.retrain:
        data_dict = retrain_data_loading(data_dict,scaler)
    return data_dict


def data_loading_compare():
    # Load and split both datasets
    df1 = load_features()
    df2 = load_features("DAX_processed.csv")
    print(df2.columns)

    # Split each dataset into train/test
    split1 = int(len(df1) * 0.8)
    train_df1, test_df1 = df1.iloc[:split1], df1.iloc[split1:]
    split2 = int(len(df2) * 0.8)
    train_df2, test_df2 = df2.iloc[:split2], df2.iloc[split2:]

    # Prepare test data (concatenate for global test set)
    X_test1 = test_df1.drop(columns=["target"]).to_numpy()
    y_test1 = test_df1["target"].to_numpy()
    X_test2 = test_df2.drop(columns=["target"]).to_numpy()
    y_test2 = test_df2["target"].to_numpy()
    X_test = np.concatenate([X_test1, X_test2], axis=0)
    y_test = np.concatenate([y_test1, y_test2], axis=0)


    # Global scaling: fit on combined train data
    X_train_all_raw = np.concatenate([
        train_df1.drop(columns=["target"]).to_numpy(),
        train_df2.drop(columns=["target"]).to_numpy()
    ], axis=0)
    scaler = StandardScaler().fit(X_train_all_raw)

    # for retrain
    X_test1_scaled = scaler.transform(X_test1)

    # Transform train and test data
    X_train1 = scaler.transform(train_df1.drop(columns=["target"]).to_numpy())
    y_train1 = train_df1["target"].to_numpy()
    X_train2 = scaler.transform(train_df2.drop(columns=["target"]).to_numpy())
    y_train2 = train_df2["target"].to_numpy()
    X_test_scaled = scaler.transform(X_test)

    # Combine for global train
    X_train_all = np.concatenate([X_train1, X_train2], axis=0)
    y_train_all = np.concatenate([y_train1, y_train2], axis=0)

    # Each client gets one full dataset
    client_slices = [
        (X_train1, y_train1),
        (X_train2, y_train2)
    ]
    if args.num_clients != 2:
        sys.exit("Error: data_loading_compare requires num_clients == 2 (one per dataset)")

    data_dict = {
        "X_train_all": X_train_all,
        "y_train_all": y_train_all,
        "X_test": X_test_scaled,
        "y_test": y_test,
        "client_slices": client_slices,
        "X_test1": X_test1_scaled,
        "y_test1": y_test1
    }
    if args.retrain:
        data_dict = retrain_data_loading(data_dict,scaler)
    return data_dict

def retrain_last_layer(fed_weights, data):
    X_train = data['X_train_retrain']
    y_train = data['y_train_retrain']
    X_test = data['X_test_retrain']
    y_test = data['y_test_retrain']

    # Model 1: Last-layer retrain (FedAvg weights, freeze all but last)
    model = build_model(X_train.shape[1])
    model.set_weights(fed_weights)
    for layer in model.layers[:-1]:
        layer.trainable = False
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    retrain_hist_loss = []
    retrain_hist_mae = []
    loss, mae = model.evaluate(X_test, y_test, verbose=0)
    retrain_hist_loss.append(loss)
    retrain_hist_mae.append(mae)
    for e in range(1, args.rounds+1):
        model.fit(X_train, y_train, epochs=1, batch_size=32, verbose=0)
        loss, mae = model.evaluate(X_test, y_test, verbose=0)
        retrain_hist_loss.append(loss)
        retrain_hist_mae.append(mae)
        print(f"[FedAvg retrain last-layer][Epoch {e:02d}] MSE={loss:.6f}, MAE={mae:.6f}")

    # Model 2: Train from scratch (no loaded weights, all layers trainable)
    scratch_model = build_model(X_train.shape[1])
    scratch_hist_loss = []
    scratch_hist_mae = []
    for e in range(1, args.rounds+1):
        scratch_model.fit(X_train, y_train, epochs=1, batch_size=32, verbose=0)
        loss, mae = scratch_model.evaluate(X_test, y_test, verbose=0)
        scratch_hist_loss.append(loss)
        scratch_hist_mae.append(mae)
        print(f"[Scratch retrain][Epoch {e:02d}] MSE={loss:.6f}, MAE={mae:.6f}")

    return retrain_hist_loss, retrain_hist_mae, scratch_hist_loss, scratch_hist_mae


def main() -> None:
    if args.compare_dataset:
        data = data_loading_compare()
    else:
        data = data_loading()
    if args.role == "server":
        # Local-only
        for cid, (Xi, yi) in enumerate(data['client_slices']):
            m = build_model(Xi.shape[1])
            local_hist_losses[cid] = []
            local_hist_maes[cid]   = []
            for e in range(1,args.rounds+1):
                if args.epoch_variance:
                    print((e-1) % (cid + 1))
                    if (e-1) % (cid + 1) == 0:
                        m.fit(Xi,yi,epochs=1,batch_size=32,verbose=0)
                else:
                    m.fit(Xi,yi,epochs=1,batch_size=32,verbose=0)
                loss,mae=m.evaluate(data['X_test'],data['y_test'],verbose=0)
                local_hist_losses[cid].append(loss)
                local_hist_maes[cid].append(mae)
                print(f"[Local][Client {cid}][Epoch {e:02d}] MSE={loss:.6f}, MAE={mae:.6f}")
        # Centralized best-epoch
        m_c = build_model(data['X_train_all'].shape[1])
        for e in range(1,args.rounds+1):
             m_c.fit(data['X_train_all'], data['y_train_all'], epochs=1, batch_size=32, verbose=0)
             loss, mae = m_c.evaluate(data['X_test'], data['y_test'], verbose=0)       
             central_hist_loss.append(loss)
             central_hist_mae.append(mae)
             print(f"[Central][Epoch {e:02d}] MSE={loss:.6f}, MAE={mae:.6f}")  

        # Federated
        print("\nStarting Federated Learning …")
        # Use patched federated server to save weights
        start_federated((data['X_test'], data['y_test']), num_rounds=args.rounds)

        retrain_hist_loss, retrain_hist_mae, scratch_hist_loss, scratch_hist_mae = None, None, None, None
        if args.retrain:
            global _best_mae_weights
            retrain_hist_loss, retrain_hist_mae, scratch_hist_loss, scratch_hist_mae = retrain_last_layer(_best_mae_weights, data)

        # Plot
        print("\n=== FINAL TEST ERRORS (round 50) ===")
        print(f"Centralised      •  MSE={central_hist_loss[-1]:.6f}  MAE={central_hist_mae[-1]:.6f}")
        for cid in range(NUM_CLIENTS):
            print(f"Local client {cid} •  MSE={local_hist_losses[cid][-1]:.6f}  MAE={local_hist_maes[cid][-1]:.6f}")
        print(f"FedAvg           •  MSE={fed_losses[-1]:.6f}  MAE={fed_maes[-1]:.6f}")
        if retrain_hist_loss is not None:
            print(f"FedAvg retrain last-layer •  MSE={retrain_hist_loss[-1]:.6f}  MAE={retrain_hist_mae[-1]:.6f}")
        if scratch_hist_loss is not None:
            print(f"FedAvg retrain from scratch •  MSE={scratch_hist_loss[-1]:.6f}  MAE={scratch_hist_mae[-1]:.6f}")
        print("\n=== BEST-EPOCH ERRORS (lowest loss anywhere in 50 rounds) ===")

        # Centralised
        best_c_idx = int(np.argmin(central_hist_loss))
        print(f"Centralised      •  Epoch {best_c_idx+1:02d}  "
         f"MSE={central_hist_loss[best_c_idx]:.6f}  "
         f"MAE={central_hist_mae [best_c_idx]:.6f}")

        # Locals
        for cid in range(NUM_CLIENTS):
            best_i = int(np.argmin(local_hist_losses[cid]))
            print(  f"Local client {cid} •  Epoch {best_i+1:02d}"
                    f"MSE={local_hist_losses[cid][best_i]:.6f}"
                    f"MAE={local_hist_maes  [cid][best_i]:.6f}")

        # FedAvg
        best_f_idx = int(np.argmin(fed_losses))
        print(  f"FedAvg           •  Round {best_f_idx+1:02d}"
                f"MSE={fed_losses[best_f_idx]:.6f}"
                f"MAE={fed_maes  [best_f_idx]:.6f}")

        # FedAvg+SP500 last-layer
        if retrain_hist_loss is not None:
            best_r_idx = int(np.argmin(retrain_hist_loss))
            print(f"FedAvg retrain last-layer •  Epoch {best_r_idx+1:02d}  "
                  f"MSE={retrain_hist_loss[best_r_idx]:.6f}  "
                  f"MAE={retrain_hist_mae[best_r_idx]:.6f}")
        if scratch_hist_loss is not None:
            best_s_idx = int(np.argmin(scratch_hist_loss))
            print(f"FedAvg retrain from scratch •  Epoch {best_s_idx+1:02d}  "
                  f"MSE={scratch_hist_loss[best_s_idx]:.6f}  "
                  f"MAE={scratch_hist_mae[best_s_idx]:.6f}")

        plot_paper_figure(local_hist_losses, local_hist_maes,
                  fed_losses, fed_maes,
                  central_hist_loss, central_hist_mae,
                  retrain_hist_loss, retrain_hist_mae,
                  scratch_hist_loss,scratch_hist_mae)
    else:
        if not (0 <= args.cid < NUM_CLIENTS):
            sys.exit("Error: cid out of range.")
        start_client(args.cid, data['client_slices'][args.cid], (data['X_test'], data['y_test']))

if __name__ == "__main__":
    main()
