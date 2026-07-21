from __future__ import annotations

import argparse
from pathlib import Path

from ids_sci.data import load_or_build_dataset, write_audit
from ids_sci.models import MODEL_NAMES
from ids_sci.reporting import build_reports
from ids_sci.runner import run_experiments


PROTOCOLS = ["random", "group", "temporal", "temporal_novel", "unseen"]


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="SCI-grade CICIDS2017 experiments")
    sub = root.add_subparsers(dest="command", required=True)

    def data_args(p):
        p.add_argument("--dataset-dir", type=Path, default=Path("dataset"))
        p.add_argument("--output-dir", type=Path, default=Path("outputs"))
        p.add_argument("--force-rebuild-cache", action="store_true")

    audit = sub.add_parser("audit", help="clean, cache, and audit CICIDS2017")
    data_args(audit)

    run = sub.add_parser("run", help="run binary robustness experiments")
    data_args(run)
    run.add_argument("--protocols", nargs="+", choices=PROTOCOLS, default=PROTOCOLS)
    run.add_argument("--models", nargs="+", choices=MODEL_NAMES,
                     default=["logistic_regression", "decision_tree", "random_forest", "xgboost"])
    run.add_argument("--seeds", nargs="+", type=int, default=[42, 52, 62])
    run.add_argument("--max-train-rows", type=int, default=600_000)
    run.add_argument("--max-test-rows", type=int, default=200_000)
    run.add_argument("--explain", action="store_true")

    report = sub.add_parser("report", help="aggregate metrics and build paper figures")
    report.add_argument("--output-dir", type=Path, default=Path("outputs"))
    return root


def load(args):
    cache_path = args.output_dir / "cache" / "cicids2017_clean.parquet"
    return load_or_build_dataset(
        args.dataset_dir,
        cache_path,
        force_rebuild=args.force_rebuild_cache,
    )


def main() -> None:
    args = parser().parse_args()
    if args.command == "report":
        build_reports(args.output_dir)
        print(f"Tables and figures written to {args.output_dir}")
        return
    loaded = load(args)
    write_audit(loaded, args.output_dir / "audit")
    if args.command == "audit":
        print(f"Dataset audit written to {args.output_dir / 'audit'}")
        return
    run_experiments(
        loaded=loaded,
        output_dir=args.output_dir,
        protocols=args.protocols,
        models=args.models,
        seeds=args.seeds,
        max_train_rows=args.max_train_rows,
        max_test_rows=args.max_test_rows,
        explain=args.explain,
    )
    build_reports(args.output_dir)
    print(f"Experiments completed. Results are in {args.output_dir}")


if __name__ == "__main__":
    main()

