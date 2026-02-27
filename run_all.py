"""
Wrapper script to run all baseline and ablation training jobs.

Skips runs whose checkpoints already exist (resume-safe).
Prints a summary table at the end.

Usage:
    python run_all.py              # Run all pending jobs
    python run_all.py --dry-run    # List jobs without executing
    python run_all.py --only baselines   # Run only baselines
    python run_all.py --only ablations   # Run only ablations
"""

import argparse
import os
import subprocess
import sys
import time


# ---------------------------------------------------------------------------
# Job definitions
# ---------------------------------------------------------------------------

BASELINE_JOBS = [
    {
        "name": "TinyCNN (track-disjoint)",
        "script": "train_baseline.py",
        "config": "configs/baselines/tiny_cnn_track.yaml",
        "checkpoint": "results/baselines/tiny_cnn_track/checkpoints/best.pt",
    },
    {
        "name": "TinyCNN (artist-disjoint)",
        "script": "train_baseline.py",
        "config": "configs/baselines/tiny_cnn_artist.yaml",
        "checkpoint": "results/baselines/tiny_cnn_artist/checkpoints/best.pt",
    },
    {
        "name": "SmallCRNN (track-disjoint)",
        "script": "train_baseline.py",
        "config": "configs/baselines/small_crnn_track.yaml",
        "checkpoint": "results/baselines/small_crnn_track/checkpoints/best.pt",
    },
    {
        "name": "SmallCRNN (artist-disjoint)",
        "script": "train_baseline.py",
        "config": "configs/baselines/small_crnn_artist.yaml",
        "checkpoint": "results/baselines/small_crnn_artist/checkpoints/best.pt",
    },
    {
        "name": "MobileNetV3 (track-disjoint)",
        "script": "train_baseline.py",
        "config": "configs/baselines/mobilenet_track.yaml",
        "checkpoint": "results/baselines/mobilenet_track/checkpoints/best.pt",
    },
    {
        "name": "MobileNetV3 (artist-disjoint)",
        "script": "train_baseline.py",
        "config": "configs/baselines/mobilenet_artist.yaml",
        "checkpoint": "results/baselines/mobilenet_artist/checkpoints/best.pt",
    },
    {
        "name": "TinyTransformer (track-disjoint)",
        "script": "train_baseline.py",
        "config": "configs/baselines/tiny_transformer_track.yaml",
        "checkpoint": "results/baselines/tiny_transformer_track/checkpoints/best.pt",
    },
    {
        "name": "TinyTransformer (artist-disjoint)",
        "script": "train_baseline.py",
        "config": "configs/baselines/tiny_transformer_artist.yaml",
        "checkpoint": "results/baselines/tiny_transformer_artist/checkpoints/best.pt",
    },
]

ABLATION_JOBS = [
    {
        "name": "Ablation: Backbone only",
        "script": "train.py",
        "config": "configs/ablations/backbone_only.yaml",
        "checkpoint": "results/ablations/backbone_only/checkpoints/best.pt",
    },
    {
        "name": "Ablation: + Early exits (no budget)",
        "script": "train.py",
        "config": "configs/ablations/exits_only.yaml",
        "checkpoint": "results/ablations/exits_only/checkpoints/best.pt",
    },
    {
        "name": "Ablation: + Exits + Budget loss",
        "script": "train.py",
        "config": "configs/ablations/exits_budget.yaml",
        "checkpoint": "results/ablations/exits_budget/checkpoints/best.pt",
    },
    {
        "name": "Ablation: Full model (GroupDRO)",
        "script": "train.py",
        "config": "configs/ablations/full_groupdro.yaml",
        "checkpoint": "results/ablations/full_groupdro/checkpoints/best.pt",
    },
]


def run_job(job, dry_run=False):
    """Run a single training job. Returns (status, elapsed_seconds)."""
    name = job["name"]
    ckpt = job["checkpoint"]

    if os.path.exists(ckpt):
        print(f"  SKIP  {name} (checkpoint exists: {ckpt})")
        return "skipped", 0

    if dry_run:
        print(f"  WOULD RUN  {name}")
        print(f"    python {job['script']} --config {job['config']}")
        return "dry_run", 0

    print(f"\n{'='*60}")
    print(f"  RUNNING  {name}")
    print(f"  Config:  {job['config']}")
    print(f"{'='*60}")

    t0 = time.time()
    result = subprocess.run(
        [sys.executable, job["script"], "--config", job["config"]],
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )
    elapsed = time.time() - t0

    if result.returncode == 0:
        print(f"  DONE  {name} ({elapsed/60:.1f} min)")
        return "completed", elapsed
    else:
        print(f"  FAILED  {name} (exit code {result.returncode})")
        return "failed", elapsed


def print_summary(results):
    """Print a summary table of all job results."""
    print(f"\n{'='*70}")
    print(f"{'TRAINING SUMMARY':^70}")
    print(f"{'='*70}")
    print(f"{'Job':<45} {'Status':<12} {'Time':>10}")
    print(f"{'-'*45} {'-'*12} {'-'*10}")

    for name, status, elapsed in results:
        time_str = f"{elapsed/60:.1f} min" if elapsed > 0 else "-"
        print(f"{name:<45} {status:<12} {time_str:>10}")

    completed = sum(1 for _, s, _ in results if s == "completed")
    skipped = sum(1 for _, s, _ in results if s == "skipped")
    failed = sum(1 for _, s, _ in results if s == "failed")
    total_time = sum(e for _, _, e in results)

    print(f"{'-'*70}")
    print(f"Completed: {completed}  Skipped: {skipped}  Failed: {failed}  "
          f"Total time: {total_time/60:.1f} min")


def main():
    parser = argparse.ArgumentParser(description="Run all training jobs")
    parser.add_argument("--dry-run", action="store_true",
                        help="List jobs without executing")
    parser.add_argument("--only", choices=["baselines", "ablations"],
                        default=None, help="Run only a subset of jobs")
    args = parser.parse_args()

    jobs = []
    if args.only is None or args.only == "baselines":
        jobs.extend(BASELINE_JOBS)
    if args.only is None or args.only == "ablations":
        jobs.extend(ABLATION_JOBS)

    print(f"Total jobs: {len(jobs)}")
    if args.dry_run:
        print("DRY RUN MODE - no training will be executed\n")

    results = []
    for job in jobs:
        status, elapsed = run_job(job, dry_run=args.dry_run)
        results.append((job["name"], status, elapsed))

    print_summary(results)


if __name__ == "__main__":
    main()
