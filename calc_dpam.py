"""
calc_dpam.py — Compute DPAM perceptual distances and append them to the
result files produced by calc_metrics_all.py.

Expected files in <run_dir>/ (parent of --purified_parent_dir):
  results_per_file.csv   ← new DPAM columns are added here (in-place)
  avg_results.txt        ← a DPAM section is appended here

DPAM is computed for:
  raw-vs-adv             (original vs adversarial, purifier-independent)
  raw-vs-prf (<name>)    (original vs purified, per purifier)
  adv-vs-prf (<name>)    (adversarial vs purified, per purifier)
"""

import sys
from argparse import ArgumentParser
from os.path import join
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm

import dpam  # available only inside the dpam container


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def mean_std(arr: np.ndarray):
    """Return (mean, std) ignoring NaNs."""
    valid = arr[~np.isnan(arr)]
    if len(valid) == 0:
        return float("nan"), float("nan")
    return float(np.mean(valid)), float(np.std(valid))


def fmt_line(label: str, mean_v: float, std_v: float, width: int) -> str:
    full = f"  {label}"
    return f"{full:<{width}} {mean_v:.3f} ± {std_v:.3f}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--original_dir", type=str, required=True,
                        help="Directory containing original (clean) audio files")
    parser.add_argument("--adversarial_dir", type=str, required=True,
                        help="Directory containing adversarial audio files")
    parser.add_argument("--purified_parent_dir", type=str, required=True,
                        help="Parent directory whose subdirectories are one purifier each "
                             "(e.g. purified/sgmse, purified/mambattention)")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Discover purifier subdirectories (same logic as calc_metrics_all.py)
    # ------------------------------------------------------------------

    purifier_dirs = sorted([
        d for d in Path(args.purified_parent_dir).iterdir()
        if d.is_dir()
    ])

    if not purifier_dirs:
        raise RuntimeError(f"No subdirectories found in {args.purified_parent_dir}")

    purifier_names = [d.name for d in purifier_dirs]
    print(f"Found {len(purifier_names)} purifier(s): {', '.join(purifier_names)}")

    # Run directory is the parent of the purified parent
    parent_dir = str(Path(args.purified_parent_dir).parent)
    results_csv = join(parent_dir, "results_per_file.csv")
    avg_path = join(parent_dir, "avg_results.txt")

    # ------------------------------------------------------------------
    # Load existing per-file CSV so we can join DPAM columns onto it
    # ------------------------------------------------------------------

    if not Path(results_csv).exists():
        print(f"ERROR: {results_csv} not found. Run calc_metrics_all.py first.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(results_csv)

    # ------------------------------------------------------------------
    # Initialise DPAM model (single instance, reused for all files)
    # ------------------------------------------------------------------

    print("Initialising DPAM model...")
    loss_fn = dpam.DPAM()

    # ------------------------------------------------------------------
    # Build column accumulators
    # ------------------------------------------------------------------

    dpam_raw_vs_adv: List[float] = []
    dpam_per_purifier: Dict[str, Dict[str, List[float]]] = {
        name: {"raw-vs-prf": [], "adv-vs-prf": []}
        for name in purifier_names
    }

    # ------------------------------------------------------------------
    # Iterate over files — driven by the filenames already in the CSV
    # so order is guaranteed to match
    # ------------------------------------------------------------------

    for filename in tqdm(df["filename"], desc="DPAM"):
        original_filename = filename.split("_")[0] + ".wav" if "dB" in filename else filename

        original_path   = join(args.original_dir,   original_filename)
        adversarial_path = join(args.adversarial_dir, filename)

        wav_original   = dpam.load_audio(original_path)
        wav_adversarial = dpam.load_audio(adversarial_path)

        # raw-vs-adv (purifier-independent)
        dpam_raw_vs_adv.append(float(loss_fn.forward(wav_original, wav_adversarial)[0]))

        # per-purifier pairs
        for name, pdir in zip(purifier_names, purifier_dirs):
            purified_path = join(str(pdir), filename)
            wav_purified = dpam.load_audio(purified_path)

            dpam_per_purifier[name]["raw-vs-prf"].append(
                float(loss_fn.forward(wav_original, wav_purified)[0])
            )
            dpam_per_purifier[name]["adv-vs-prf"].append(
                float(loss_fn.forward(wav_adversarial, wav_purified)[0])
            )

    # ------------------------------------------------------------------
    # Attach new columns to the DataFrame (overwrite if already present)
    # ------------------------------------------------------------------

    df["dpam_raw-vs-adv"] = dpam_raw_vs_adv
    for name in purifier_names:
        for pair in ["raw-vs-prf", "adv-vs-prf"]:
            df[f"dpam_{pair}_{name}"] = dpam_per_purifier[name][pair]

    df.to_csv(results_csv, index=False)
    print(f"DPAM columns appended to: {results_csv}")

    # ------------------------------------------------------------------
    # Compute averages and build the summary block
    # ------------------------------------------------------------------

    all_labels = (
        ["raw-vs-adv"]
        + [f"{pair} ({n})" for n in purifier_names for pair in ["raw-vs-prf", "adv-vs-prf"]]
    )
    W = max(len(lbl) for lbl in all_labels) + 4

    lines = ["\nDPAM:"]

    mean_v, std_v = mean_std(np.array(dpam_raw_vs_adv))
    lines.append(fmt_line("raw-vs-adv", mean_v, std_v, W))

    for name in purifier_names:
        for pair in ["raw-vs-prf", "adv-vs-prf"]:
            mean_v, std_v = mean_std(np.array(dpam_per_purifier[name][pair]))
            lines.append(fmt_line(f"{pair} ({name})", mean_v, std_v, W))

    dpam_block = "\n".join(lines) + "\n"
    print(dpam_block)

    # ------------------------------------------------------------------
    # Append DPAM block to avg_results.txt
    # Strip any existing DPAM section first so re-runs don't duplicate it
    # ------------------------------------------------------------------

    if Path(avg_path).exists():
        with open(avg_path, "r") as f:
            existing = f.read()
        # Remove a previously appended DPAM section if present
        dpam_marker = "\nDPAM:"
        if dpam_marker in existing:
            existing = existing[: existing.index(dpam_marker)]
        existing = existing.rstrip()
    else:
        existing = ""

    with open(avg_path, "w") as f:
        f.write(existing + "\n" + dpam_block)

    print(f"DPAM averages appended to: {avg_path}")
