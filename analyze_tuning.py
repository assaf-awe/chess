#!/usr/bin/env python3
import argparse
import csv
import json
import re
from datetime import datetime
from pathlib import Path

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
VAL_RE = re.compile(
    r"VAL:\s*(\d+)\.\s*mLoss:([0-9.]+)\s*\|\s*mAcc:([0-9.]+)%\s*\|\s*wAcc:([0-9.]+)%"
)


def parse_log(log_path: Path):
    last = None
    with log_path.open("r", errors="ignore") as f:
        for line in f:
            clean = ANSI_RE.sub("", line)
            match = VAL_RE.search(clean)
            if match:
                last = {
                    "val_epoch": int(match.group(1)),
                    "mLoss": float(match.group(2)),
                    "mAcc": float(match.group(3)),
                    "wAcc": float(match.group(4)),
                }
    return last


def load_meta(meta_path: Path):
    if not meta_path:
        return {}
    meta = {}
    with meta_path.open("r", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            run_id = row.get("run_id")
            if run_id:
                meta[run_id] = row
    return meta


def collect_results(log_dir: Path, meta_path: Path | None):
    logs = sorted(log_dir.rglob("*.log"))
    meta = load_meta(meta_path) if meta_path else {}
    results = []
    for log_path in logs:
        run_id = log_path.stem
        metrics = parse_log(log_path)
        if not metrics:
            continue
        row = {
            "run_id": run_id,
            "log_path": str(log_path),
            **metrics,
        }
        row.update(meta.get(run_id, {}))
        results.append(row)
    return results


def sort_key(metric):
    if metric == "mLoss":
        return lambda r: (r.get(metric, float("inf")), r.get("wAcc", -1.0))
    return lambda r: (-r.get(metric, -1.0), -r.get("mAcc", -1.0))


def write_summary(results, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "summary.csv"
    md_path = out_dir / "summary.md"

    fieldnames = [
        "run_id",
        "phase",
        "epochs",
        "lr",
        "weight_decay",
        "do_f",
        "d_model",
        "nhead",
        "num_layers",
        "batch_size",
        "optim",
        "compile",
        "dataset",
        "num_workers",
        "val_epoch",
        "mLoss",
        "mAcc",
        "wAcc",
        "log_path",
        "settings_path",
        "save_prefix",
    ]

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    by_wacc = sorted(results, key=sort_key("wAcc"))
    by_phase = {}
    for row in results:
        phase = row.get("phase", "unknown")
        best = by_phase.get(phase)
        if not best or row.get("wAcc", -1) > best.get("wAcc", -1):
            by_phase[phase] = row

    with md_path.open("w") as f:
        f.write("# Tuning summary\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Top runs by wAcc (last VAL line)\n")
        for row in by_wacc[:5]:
            f.write(
                "- "
                f"{row.get('run_id')} | "
                f"wAcc={row.get('wAcc')}% | "
                f"mAcc={row.get('mAcc')}% | "
                f"mLoss={row.get('mLoss')} | "
                f"lr={row.get('lr')} | "
                f"wd={row.get('weight_decay')} | "
                f"do_f={row.get('do_f')} | "
                f"epochs={row.get('epochs')} | "
                f"phase={row.get('phase')}\n"
            )
        f.write("\n## Best run per phase\n")
        for phase, row in sorted(by_phase.items()):
            f.write(
                "- "
                f"{phase}: {row.get('run_id')} | "
                f"wAcc={row.get('wAcc')}% | "
                f"lr={row.get('lr')} | "
                f"wd={row.get('weight_decay')} | "
                f"do_f={row.get('do_f')} | "
                f"epochs={row.get('epochs')}\n"
            )
        f.write("\n## Notes\n")
        f.write("- Metrics are taken from the last VAL line of each run log.\n")
        f.write("- Use summary.csv for sorting/filtering across all runs.\n")


def main():
    parser = argparse.ArgumentParser(description="Analyze tuning logs and report.")
    parser.add_argument("--log-dir", required=True, help="Directory containing run logs.")
    parser.add_argument("--meta", help="Path to meta.jsonl")
    parser.add_argument("--out-dir", help="Directory to write summary.csv/.md")
    parser.add_argument("--best-metric", choices=["wAcc", "mAcc", "mLoss"], help="Metric for best selection.")
    parser.add_argument("--top-n", type=int, default=1, help="Number of top runs to output.")
    parser.add_argument("--format", choices=["text", "json"], default="text", help="Output format for selections.")
    parser.add_argument("--print-key", help="Print a single field for the best run.")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    meta_path = Path(args.meta) if args.meta else None

    results = collect_results(log_dir, meta_path)
    if not results:
        raise SystemExit("No valid runs found in logs.")

    if args.best_metric:
        ranked = sorted(results, key=sort_key(args.best_metric))
        bests = ranked[: args.top_n]
        if args.print_key:
            if args.top_n == 1:
                print(bests[0].get(args.print_key, ""))
            else:
                for row in bests:
                    print(row.get(args.print_key, ""))
        elif args.format == "json":
            print(json.dumps(bests, indent=2))
        else:
            for row in bests:
                print(row.get("run_id", ""))

    if args.out_dir:
        write_summary(sorted(results, key=sort_key("wAcc")), Path(args.out_dir))


if __name__ == "__main__":
    main()
