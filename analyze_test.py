#!/usr/bin/env python3
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Improve plot aesthetics
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'legend.framealpha': 0.9,
    'legend.edgecolor': 'lightgray',
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 100,
})

# Color palette for models
MODEL_COLORS = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6', '#f39c12', '#1abc9c', '#e67e22', '#34495e']


def clean_legend_name(name: str) -> str:
    """Remove 'pred_' prefix from legend names for cleaner display."""
    if name.startswith("pred_"):
        return name[5:]
    return name


def display_name(name: str, legend_map: dict) -> str:
    """Resolve legend name with overrides and cleanup."""
    if name in legend_map:
        return legend_map[name]
    return clean_legend_name(name)

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from chess_models import MyChessNet, MyChessNet_trnsfrm2_do, MyChessNet_hist


WINNER_MAP = {"white": 1, "black": -1, "draw": 0}
OUTCOME_ORDER = [-1, 0, 1]
OUTCOME_LABELS = {-1: "loss", 0: "draw", 1: "win"}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_dataframe(paths):
    frames = []
    for path in paths:
        with open(path, "rb") as f:
            frames.append(pd.read_pickle(f))
    return pd.concat(frames, ignore_index=True)


def compute_labels(df: pd.DataFrame) -> np.ndarray:
    winners = df["winner"].map(WINNER_MAP).fillna(0).astype(int).to_numpy()
    turn = df["turn"].to_numpy()
    return np.where(turn % 2 == 0, -winners, winners)


def build_eval_predictions(df: pd.DataFrame, threshold_cp: float) -> np.ndarray:
    eval_cp = df["eval_cp"].to_numpy()
    eval_mate = df["eval_mate"].to_numpy()
    is_white_turn = (df["turn"].to_numpy() % 2) == 1
    sign = np.where(is_white_turn, 1.0, -1.0)

    eval_cp_stm = np.where(np.isnan(eval_cp), 0.0, eval_cp) * sign
    eval_mate_stm = np.where(np.isnan(eval_mate), 0.0, eval_mate) * sign
    has_mate = ~np.isnan(eval_mate)

    preds = np.zeros(len(df), dtype=int)
    preds[has_mate] = np.where(eval_mate_stm[has_mate] > 0, 1, -1)

    no_mate = ~has_mate
    preds[no_mate] = np.where(
        eval_cp_stm[no_mate] > threshold_cp,
        1,
        np.where(eval_cp_stm[no_mate] < -threshold_cp, -1, 0),
    )
    return preds


def calibrate_eval_threshold(df: pd.DataFrame, labels: np.ndarray, grid):
    best = {"threshold": None, "acc": -1.0}
    for threshold in grid:
        preds = build_eval_predictions(df, threshold)
        acc = (preds == labels).mean()
        if acc > best["acc"]:
            best = {"threshold": threshold, "acc": acc}
    return best


class EvalStateDataset(Dataset):
    def __init__(self, df: pd.DataFrame, labels: np.ndarray):
        self.matrices = np.stack(df["matrix"].values)
        self.turns = df["turn"].to_numpy()
        self.white_castling = np.stack(df["white_castling"].values).astype(np.float32)
        self.black_castling = np.stack(df["black_castling"].values).astype(np.float32)
        self.labels = labels.astype(np.int64)

    def __len__(self):
        return len(self.matrices)

    def __getitem__(self, idx):
        brd_state = self.matrices[idx]
        turn = int(self.turns[idx])

        if turn % 2 == 0:  # black turn
            brd_state = np.flipud(-brd_state)

        n = torch.tensor(brd_state + 7, dtype=torch.int64).unsqueeze(0)
        brd_state3d = torch.zeros(15, 8, 8, dtype=torch.float32)
        brd_state3d.scatter_(0, n, 1.0)
        brd_state3d = brd_state3d[[0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14], :, :]

        if turn % 2 == 1:  # white turn
            brd_state3d[-1, 7, 0] = float(self.white_castling[idx][0])
            brd_state3d[-1, 7, 7] = float(self.white_castling[idx][1])
            brd_state3d[0, 0, 0] = float(self.black_castling[idx][0])
            brd_state3d[0, 0, 7] = float(self.black_castling[idx][1])
        else:
            brd_state3d[-1, 7, 0] = float(self.black_castling[idx][0])
            brd_state3d[-1, 7, 7] = float(self.black_castling[idx][1])
            brd_state3d[0, 0, 0] = float(self.white_castling[idx][0])
            brd_state3d[0, 0, 7] = float(self.white_castling[idx][1])

        return brd_state3d, self.labels[idx]


def infer_model(model_path: Path):
    stem = model_path.stem
    if "hist" in stem:
        return MyChessNet_hist()
    if "cnn" in stem:
        return MyChessNet()
    if "trnsfrm2" in stem:
        return MyChessNet_trnsfrm2_do(d_model=128, nhead=16, num_layers=6, do_f=1)
    raise ValueError(f"Unknown model type for {model_path}")


def run_model_predictions(model_path: Path, dataset: Dataset, batch_size: int, device: str):
    model = infer_model(model_path)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    model.to(device)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    preds = np.zeros(len(dataset), dtype=np.int64)
    offset = 0
    with torch.no_grad():
        for features, _labels in loader:
            features = features.to(device)
            logits = model(features)
            batch_preds = torch.argmax(logits, dim=1).cpu().numpy() - 1
            preds[offset : offset + len(batch_preds)] = batch_preds
            offset += len(batch_preds)
    return preds


def save_hist(series, out_path, title, xlabel, bins=50, log=False, smooth=False, smooth_window=9):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    if smooth:
        counts, edges = np.histogram(series, bins=bins, density=True)
        centers = (edges[:-1] + edges[1:]) / 2
        window = max(3, int(smooth_window) | 1)
        kernel = np.ones(window, dtype=float) / window
        smooth_counts = np.convolve(counts, kernel, mode="same")
        ax.plot(centers, smooth_counts, color='#3498db', linewidth=2.2)
        ax.fill_between(centers, smooth_counts, color='#3498db', alpha=0.25)
    else:
        ax.hist(series, bins=bins, density=True, color='#3498db', edgecolor='white', linewidth=0.5, alpha=0.85)
    if log:
        ax.set_yscale("log")
    ax.set_title(title, pad=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight', dpi=120)
    plt.close(fig)


def save_bar(values, out_path, title, xlabel, ylabel="Count"):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = [MODEL_COLORS[i % len(MODEL_COLORS)] for i in range(len(values))]
    values.plot(kind="bar", ax=ax, color=colors, edgecolor='white', linewidth=0.5)
    ax.set_title(title, pad=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis='x', rotation=30)
    for label in ax.get_xticklabels():
        label.set_ha('right')
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight', dpi=120)
    plt.close(fig)


def save_accuracy_bar(names, values, display_names, out_path, title):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = [MODEL_COLORS[i % len(MODEL_COLORS)] for i in range(len(values))]
    x_pos = np.arange(len(names))
    bars = ax.bar(x_pos, values, color=colors, edgecolor='white', linewidth=0.5)
    ax.set_ylim(0, 1)
    ax.set_title(title, pad=12)
    ax.set_ylabel("Accuracy")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(display_names, rotation=30, ha="right")
    # Add value labels on bars
    for bar, val in zip(bars, values):
        ax.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width() / 2, val),
                    xytext=(0, 4), textcoords='offset points',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight', dpi=120)
    plt.close(fig)


def format_bin_label(interval, as_int=False):
    """Format interval labels, optionally as integers."""
    left = interval.left
    right = interval.right
    if as_int:
        left = int(round(left))
        right = int(round(right))
        return f"({left}, {right}]"
    return str(interval)


def accuracy_by_bins(df, label_col, pred_cols, feature, bins, right=True, int_labels=False):
    cut = pd.cut(df[feature], bins=bins, right=right, include_lowest=True)
    if int_labels:
        bin_labels = [format_bin_label(iv, as_int=True) for iv in cut.cat.categories]
    else:
        bin_labels = cut.cat.categories.astype(str)
    result = {"bin": bin_labels}
    for pred_name in pred_cols:
        grouped = df.groupby(cut, observed=True)[[label_col, pred_name]].apply(
            lambda g: (g[label_col] == g[pred_name]).mean()
        )
        result[pred_name] = grouped.to_numpy()
    counts = df.groupby(cut, observed=True)[label_col].size().to_numpy()
    result["count"] = counts
    return pd.DataFrame(result)


def accuracy_by_category(df, label_col, pred_cols, feature):
    grouped = df.groupby(feature, observed=True)
    rows = []
    for value, group in grouped:
        row = {"category": value, "count": len(group)}
        for pred_name in pred_cols:
            row[pred_name] = (group[label_col] == group[pred_name]).mean()
        rows.append(row)
    return pd.DataFrame(rows).sort_values("count", ascending=False)


def save_accuracy_lines(acc_df, out_path, title, xlabel, pred_cols, display_names):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(acc_df))
    for i, pred_name in enumerate(pred_cols):
        color = MODEL_COLORS[i % len(MODEL_COLORS)]
        ax.plot(x, acc_df[pred_name], marker="o", label=display_names[i],
                color=color, linewidth=2, markersize=6)
    ax.set_xticks(x)
    ax.set_xticklabels(acc_df["bin"], rotation=45, ha="right")
    ax.set_title(title, pad=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.legend(loc='best', frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight', dpi=120)
    plt.close(fig)


def save_accuracy_category(acc_df, out_path, title, xlabel, pred_cols, display_names):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(acc_df))
    for i, pred_name in enumerate(pred_cols):
        color = MODEL_COLORS[i % len(MODEL_COLORS)]
        ax.plot(x, acc_df[pred_name], marker="o", label=display_names[i],
                color=color, linewidth=2, markersize=6)
    ax.set_xticks(x)
    ax.set_xticklabels(acc_df["category"], rotation=45, ha="right")
    ax.set_title(title, pad=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.legend(loc='best', frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight', dpi=120)
    plt.close(fig)

def predicted_distribution_by_bins(df, pred_col, feature, bins, right=True, int_labels=False):
    cut = pd.cut(df[feature], bins=bins, right=right, include_lowest=True)
    rows = []
    for bin_val, group in df.groupby(cut, observed=True):
        counts = group[pred_col].value_counts().reindex(OUTCOME_ORDER, fill_value=0)
        total = counts.sum()
        rows.append(
            {
                "bin": format_bin_label(bin_val, as_int=int_labels) if int_labels else str(bin_val),
                "count": int(total),
                "loss": int(counts.get(-1, 0)),
                "draw": int(counts.get(0, 0)),
                "win": int(counts.get(1, 0)),
            }
        )
    return pd.DataFrame(rows)

def predicted_distribution_by_category(df, pred_col, feature):
    rows = []
    for value, group in df.groupby(feature, observed=True):
        counts = group[pred_col].value_counts().reindex(OUTCOME_ORDER, fill_value=0)
        total = counts.sum()
        rows.append(
            {
                "category": value,
                "count": int(total),
                "loss": int(counts.get(-1, 0)),
                "draw": int(counts.get(0, 0)),
                "win": int(counts.get(1, 0)),
            }
        )
    return pd.DataFrame(rows).sort_values("count", ascending=False)

def save_pred_distribution(dist_df, out_path, title, xlabel, category_col):
    if dist_df.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(dist_df))
    bottom = np.zeros(len(dist_df))
    totals = dist_df["count"].replace(0, np.nan).to_numpy(dtype=float)
    outcome_colors = {'loss': '#e74c3c', 'draw': '#95a5a6', 'win': '#2ecc71'}
    for outcome in ["loss", "draw", "win"]:
        values = dist_df[outcome].to_numpy(dtype=float) / totals
        values = np.nan_to_num(values)
        ax.bar(x, values, bottom=bottom, label=outcome, color=outcome_colors[outcome],
               edgecolor='white', linewidth=0.5)
        bottom += values
    ax.set_xticks(x)
    ax.set_xticklabels(dist_df[category_col], rotation=45, ha="right")
    ax.set_title(title, pad=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Prediction share")
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right', frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight', dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Analyze test dataset and model accuracy.")
    parser.add_argument(
        "--dataset",
        nargs="+",
        required=True,
        help="Path(s) to test pickle(s)",
    )
    parser.add_argument("--out-dir", default=None, help="Output directory for plots")
    parser.add_argument(
        "--models",
        nargs="*",
        default=[
            "models/trnsfrm2_do_lichess499k_deeper_test_18.pth",
            "models/hist_lichess499k_test_19.pth",
        ],
        help="Model checkpoint paths",
    )
    parser.add_argument(
        "--legends",
        nargs="*",
        default=None,
        help="Legend labels for models (same order as --models)",
    )
    parser.add_argument(
        "--skip-models",
        action="store_true",
        help="Skip model inference and reuse pred_* columns from dataset",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--eval-threshold", type=float, default=50.0)
    parser.add_argument(
        "--calibrate-eval-threshold",
        action="store_true",
        help="Find best eval threshold on the dataset",
    )
    parser.add_argument(
        "--eval-threshold-grid",
        default="0,400,10",
        help="Grid for calibration: start,stop,step (centipawns)",
    )
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()

    df = load_dataframe([Path(p) for p in args.dataset])
    if args.max_rows:
        df = df.sample(args.max_rows, random_state=0).reset_index(drop=True)

    labels = compute_labels(df)
    df = df.copy()
    df["label_stm"] = labels

    out_dir = Path(args.out_dir) if args.out_dir else Path(
        "analysis/test_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    plots_dir = ensure_dir(out_dir / "plots")
    data_dir = ensure_dir(plots_dir / "data")
    perf_dir = ensure_dir(plots_dir / "performance")
    dist_dir = ensure_dir(perf_dir / "predicted_distributions")

    eval_threshold = args.eval_threshold
    calibrated_threshold = None
    if args.calibrate_eval_threshold:
        start, stop, step = (float(x) for x in args.eval_threshold_grid.split(","))
        grid = np.arange(start, stop + step, step)
        best = calibrate_eval_threshold(df, labels, grid)
        calibrated_threshold = best["threshold"]

    pred_columns = {}
    eval_preds = build_eval_predictions(df, eval_threshold)
    df["pred_eval_threshold"] = eval_preds
    pred_columns["eval_threshold"] = "pred_eval_threshold"

    if calibrated_threshold is not None:
        df["pred_eval_calibrated"] = build_eval_predictions(df, calibrated_threshold)
        pred_columns["eval_calibrated"] = "pred_eval_calibrated"

    model_preds = {}
    legend_map = {"eval_threshold": "Stockfish", "eval_calibrated": "Stockfish"}
    model_legend_map = {}
    if args.legends is not None:
        if not args.models or len(args.legends) != len(args.models):
            raise ValueError("--legends must match the number of --models")
        model_legend_map = {
            Path(model_path).stem: legend
            for model_path, legend in zip(args.models, args.legends)
        }
        legend_map.update(model_legend_map)

    if args.models:
        dataset = EvalStateDataset(df, labels)
        for model_path in args.models:
            model_path = Path(model_path)
            key = model_path.stem
            pred_col = f"pred_{key}"
            if args.skip_models:
                if pred_col not in df.columns:
                    raise ValueError(
                        f"Missing {pred_col} in dataset for --skip-models"
                    )
            else:
                preds = run_model_predictions(model_path, dataset, args.batch_size, args.device)
                df[pred_col] = preds
                model_preds[key] = preds
            pred_columns[key] = pred_col

    pred_col_names = list(pred_columns.values())
    pred_display_names = [display_name(name, legend_map) for name in pred_columns.keys()]

    # Save predictions for plot-only reruns
    df.to_pickle(out_dir / "predictions.pkl")

    # Data-only plots
    save_bar(df["winner"].value_counts(), data_dir / "outcomes.png", "Winner counts", "winner")
    victory_status_freq = df["victory_status"].value_counts(normalize=True)
    save_bar(
        victory_status_freq,
        data_dir / "victory_status.png",
        "Termination reason",
        "victory_status",
        ylabel="Frequency",
    )
    save_hist(df["rating_mean"].dropna(), data_dir / "rating_mean.png", "Rating mean", "rating_mean")
    save_hist(df["rating_diff"].dropna(), data_dir / "rating_diff.png", "Rating diff", "rating_diff")
    save_bar(
        df["time_control_class"].fillna("unknown").value_counts(),
        data_dir / "time_control_class.png",
        "Time control class",
        "time_control_class",
    )
    save_hist(
        df["time_control_base"].dropna(),
        data_dir / "time_control_base.png",
        "Time control base (sec)",
        "time_control_base",
    )
    save_hist(
        df["time_control_inc"].dropna(),
        data_dir / "time_control_inc.png",
        "Time control increment (sec)",
        "time_control_inc",
    )
    save_hist(
        df["num_ply"].dropna(),
        data_dir / "num_ply.png",
        "Num ply",
        "num_ply",
        bins=120,
        smooth=True,
        smooth_window=11,
    )
    save_hist(df["turn"].dropna(), data_dir / "turn.png", "Ply (turn)", "turn")
    save_hist(
        df["clock_sec"].dropna(),
        data_dir / "clock_sec.png",
        "Clock seconds",
        "clock_sec",
    )
    eval_cp = df["eval_cp"].dropna()
    if not eval_cp.empty:
        clipped = eval_cp.clip(-1000, 1000)
        save_hist(clipped, data_dir / "eval_cp.png", "Eval (cp) clipped", "eval_cp")
        save_hist(
            clipped.abs(),
            data_dir / "eval_abs_cp.png",
            "Eval |cp| clipped",
            "abs(eval_cp)",
        )
    bot_counts = pd.Series(
        {
            "white_is_bot": df["white_is_bot"].sum(),
            "black_is_bot": df["black_is_bot"].sum(),
        }
    )
    save_bar(bot_counts, data_dir / "bot_counts.png", "Bot flags", "bot_flag")

    # Overall accuracy
    overall_values = [(df[col] == df["label_stm"]).mean() for col in pred_columns.values()]
    overall_names = list(pred_columns.keys())
    overall_acc = dict(zip(overall_names, overall_values))
    save_accuracy_bar(
        overall_names,
        overall_values,
        pred_display_names,
        perf_dir / "accuracy_overall.png",
        "Overall accuracy",
    )

    # Accuracy by ply
    df_turn = df[df["turn"].notna()].copy()
    max_ply = min(int(df_turn["turn"].max()), 200)
    df_turn = df_turn[df_turn["turn"] <= max_ply]
    ply_bins = np.linspace(1, max_ply, 12).astype(int)
    ply_bins = np.unique(ply_bins)  # Remove duplicates from rounding
    ply_acc = accuracy_by_bins(df_turn, "label_stm", pred_col_names, "turn", ply_bins, int_labels=True)
    ply_acc.to_csv(out_dir / "accuracy_by_ply.csv", index=False)
    save_accuracy_lines(
        ply_acc,
        perf_dir / "accuracy_by_ply.png",
        "Accuracy by ply",
        "Ply bin",
        pred_col_names,
        pred_display_names,
    )
    actual_outcomes_by_ply = predicted_distribution_by_bins(
        df_turn, "label_stm", "turn", ply_bins, int_labels=True
    )
    save_pred_distribution(
        actual_outcomes_by_ply,
        dist_dir / "outcomes_by_ply.png",
        "Actual outcomes by ply",
        "Ply bin",
        "bin",
    )
    for name, pred_col in pred_columns.items():
        dist = predicted_distribution_by_bins(df_turn, pred_col, "turn", ply_bins, int_labels=True)
        save_pred_distribution(
            dist,
            dist_dir / f"pred_dist_{name}_by_ply.png",
            f"Predicted outcomes by ply ({name})",
            "ply bin",
            "bin",
        )

    # Accuracy by rating diff
    df_diff = df[df["rating_diff"].notna()].copy()
    diff_bins = [-400, -200, -100, -50, 0, 50, 100, 200, 400]
    diff_acc = accuracy_by_bins(df_diff, "label_stm", pred_col_names, "rating_diff", diff_bins)
    diff_acc.to_csv(out_dir / "accuracy_by_rating_diff.csv", index=False)
    save_accuracy_lines(
        diff_acc,
        perf_dir / "accuracy_by_rating_diff.png",
        "Accuracy by rating diff",
        "Rating diff bin",
        pred_col_names,
        pred_display_names,
    )
    for name, pred_col in pred_columns.items():
        dist = predicted_distribution_by_bins(df_diff, pred_col, "rating_diff", diff_bins)
        save_pred_distribution(
            dist,
            dist_dir / f"pred_dist_{name}_by_rating_diff.png",
            f"Predicted outcomes by rating diff ({name})",
            "rating_diff bin",
            "bin",
        )

    # Accuracy by rating mean
    df_mean = df[df["rating_mean"].notna()].copy()
    mean_bins = [0, 800, 1200, 1600, 2000, 2400, 2800, 3200]
    mean_acc = accuracy_by_bins(df_mean, "label_stm", pred_col_names, "rating_mean", mean_bins)
    mean_acc.to_csv(out_dir / "accuracy_by_rating_mean.csv", index=False)
    save_accuracy_lines(
        mean_acc,
        perf_dir / "accuracy_by_rating_mean.png",
        "Accuracy by rating mean",
        "Rating mean bin",
        pred_col_names,
        pred_display_names,
    )
    for name, pred_col in pred_columns.items():
        dist = predicted_distribution_by_bins(df_mean, pred_col, "rating_mean", mean_bins)
        save_pred_distribution(
            dist,
            dist_dir / f"pred_dist_{name}_by_rating_mean.png",
            f"Predicted outcomes by rating mean ({name})",
            "rating_mean bin",
            "bin",
        )

    # Accuracy by time control class
    tc_acc = accuracy_by_category(
        df.assign(time_control_class=df["time_control_class"].fillna("unknown")),
        "label_stm",
        pred_col_names,
        "time_control_class",
    )
    tc_acc.to_csv(out_dir / "accuracy_by_time_control.csv", index=False)
    save_accuracy_category(
        tc_acc,
        perf_dir / "accuracy_by_time_control.png",
        "Accuracy by time control class",
        "Time control class",
        pred_col_names,
        pred_display_names,
    )
    df_tc = df.assign(time_control_class=df["time_control_class"].fillna("unknown"))
    for name, pred_col in pred_columns.items():
        dist = predicted_distribution_by_category(df_tc, pred_col, "time_control_class")
        save_pred_distribution(
            dist,
            dist_dir / f"pred_dist_{name}_by_time_control.png",
            f"Predicted outcomes by time control ({name})",
            "time_control_class",
            "category",
        )

    # Accuracy by victory status
    vs_acc = accuracy_by_category(
        df.assign(victory_status=df["victory_status"].fillna("unknown")),
        "label_stm",
        pred_col_names,
        "victory_status",
    )
    vs_acc.to_csv(out_dir / "accuracy_by_victory_status.csv", index=False)
    save_accuracy_category(
        vs_acc,
        perf_dir / "accuracy_by_victory_status.png",
        "Accuracy by termination reason",
        "Termination reason",
        pred_col_names,
        pred_display_names,
    )
    df_vs = df.assign(victory_status=df["victory_status"].fillna("unknown"))
    for name, pred_col in pred_columns.items():
        dist = predicted_distribution_by_category(df_vs, pred_col, "victory_status")
        save_pred_distribution(
            dist,
            dist_dir / f"pred_dist_{name}_by_victory_status.png",
            f"Predicted outcomes by termination ({name})",
            "victory_status",
            "category",
        )

    # Accuracy by eval strength
    if not eval_cp.empty:
        df_eval = df[df["eval_cp"].notna()].copy()
        df_eval["eval_abs_cp"] = df_eval["eval_cp"].abs()
        eval_bins = [0, 50, 100, 200, 400, 800, 1200]
        eval_acc = accuracy_by_bins(df_eval, "label_stm", pred_col_names, "eval_abs_cp", eval_bins)
        eval_acc.to_csv(out_dir / "accuracy_by_eval_abs.csv", index=False)
        save_accuracy_lines(
            eval_acc,
            perf_dir / "accuracy_by_eval_abs.png",
            "Accuracy by eval |cp|",
            "Eval |cp| bin",
            pred_col_names,
            pred_display_names,
        )
        for name, pred_col in pred_columns.items():
            dist = predicted_distribution_by_bins(df_eval, pred_col, "eval_abs_cp", eval_bins)
            save_pred_distribution(
                dist,
                dist_dir / f"pred_dist_{name}_by_eval_abs.png",
                f"Predicted outcomes by eval |cp| ({name})",
                "eval_abs_cp bin",
                "bin",
            )

    # Accuracy by clock seconds
    if df["clock_sec"].notna().any():
        df_clock = df[df["clock_sec"].notna()].copy()
        clock_bins = [0, 5, 10, 20, 30, 60, 120, 300]
        clock_acc = accuracy_by_bins(df_clock, "label_stm", pred_col_names, "clock_sec", clock_bins)
        clock_acc.to_csv(out_dir / "accuracy_by_clock.csv", index=False)
        save_accuracy_lines(
            clock_acc,
            perf_dir / "accuracy_by_clock.png",
            "Accuracy by clock seconds",
            "Clock seconds bin",
            pred_col_names,
            pred_display_names,
        )
        for name, pred_col in pred_columns.items():
            dist = predicted_distribution_by_bins(df_clock, pred_col, "clock_sec", clock_bins)
            save_pred_distribution(
                dist,
                dist_dir / f"pred_dist_{name}_by_clock.png",
                f"Predicted outcomes by clock ({name})",
                "clock_sec bin",
                "bin",
            )

    metrics = {
        "dataset_rows": len(df),
        "eval_threshold": eval_threshold,
        "eval_calibrated_threshold": calibrated_threshold,
        "overall_accuracy": overall_acc,
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
