# Written by Alvin Yadav

"""
Train/refresh the dom_xmax predictor from existing batch runs, without going
through run_tests.py.

Two uses:

1. Backfill historical batches. Point at one or more directories that each
   contain a `batch_configs/*.json` folder and a
   `batch_results/test_results.json` (the layout produced by run_tests.py —
   e.g. AEM_Batch_runs/, or any archived batch). The pre-rename names
   `configs_test_batch/` and `results_test_batch/` are still accepted, so
   existing archives ingest unchanged. Every DONE row is turned into a
   labelled dataset row (features + L_max + censored) and appended to the
   shared dataset, deduped by config content hash so re-running this is
   always safe.

       python train_xmax_from_runs.py ../AEM_Batch_runs

2. Resync + refit only. Run with no directory arguments to just refit the
   local model from whatever is currently in the shared dataset — the thing
   to do after `git pull` brings in rows another machine collected.

       python train_xmax_from_runs.py

Historical runs from before this predictor existed don't have
dom_xmax_final/n_extensions/censored recorded. For those, censoring is
inferred from the extension count in phase_timings: a run that used fewer
than the extension cap in effect at the time is provably not capped (the
domain-extension loop only stops early when the plume fits); a run that hit
the cap is ambiguous and is conservatively marked censored (excluded from
training, kept in the dataset for audit). Pass --max-extensions-at-run-time
if a given batch was produced with a non-default MAX_EXTENSIONS.
"""

import argparse
import json
import os
import sys

from at_config import ATConfiguration
import xmax_features
from xmax_predictor import XmaxPredictor, DEFAULT_DATASET_PATH, DEFAULT_MODEL_PATH, load_dataset_rows

# MAX_EXTENSIONS in at_simulation.py before it was raised to 6 (2026-07-06).
# Used as the default assumption when backfilling old test_results.json
# files that predate n_extensions/censored instrumentation.
LEGACY_MAX_EXTENSIONS = 3


# run_tests.py's output directories were renamed (configs_test_batch ->
# batch_configs, results_test_batch -> batch_results). Archived batches on
# disk still use the old names, so both are accepted when ingesting.
_CONFIG_DIR_NAMES = ("batch_configs", "configs_test_batch")
_RESULT_DIR_NAMES = ("batch_results", "results_test_batch")


def _find_configs_dir(batch_dir):
    for name in _CONFIG_DIR_NAMES:
        cand = os.path.join(batch_dir, name)
        if os.path.isdir(cand) and any(f.endswith(".json") for f in os.listdir(cand)):
            return cand
    if os.path.isdir(batch_dir) and any(f.endswith(".json") for f in os.listdir(batch_dir)):
        return batch_dir
    raise FileNotFoundError(f"No *.json configs found under {batch_dir}")


def _find_results_json(batch_dir):
    cands = [os.path.join(batch_dir, name, "test_results.json")
             for name in _RESULT_DIR_NAMES]
    cands.append(os.path.join(batch_dir, "test_results.json"))
    for cand in cands:
        if os.path.isfile(cand):
            return cand
    raise FileNotFoundError(f"No test_results.json found under {batch_dir}")


def ingest_batch_dir(batch_dir, max_extensions_at_run_time=LEGACY_MAX_EXTENSIONS):
    """
    Turn one historical batch directory into a list of dataset-row dicts.

    Returns (rows, n_skipped_missing_config).
    """
    configs_dir = _find_configs_dir(batch_dir)
    results_path = _find_results_json(batch_dir)

    with open(results_path) as f:
        results = json.load(f)

    rows = []
    n_skipped = 0
    for r in results:
        if r.get("status") != "DONE":
            continue

        cfg_path = os.path.join(configs_dir, r["config"])
        if not os.path.isfile(cfg_path):
            n_skipped += 1
            continue

        config = ATConfiguration.from_json(cfg_path)
        features = xmax_features.extract_features(config)
        config_hash = xmax_features.config_hash_from_path(cfg_path)

        if "censored" in r and "n_extensions" in r:
            # Already-instrumented row (produced by the current run_tests.py).
            n_extensions = r["n_extensions"]
            censored = r["censored"]
            dom_xmax_initial = r.get("dom_xmax_initial", config.dom_xmax)
            dom_xmax_final = r.get("dom_xmax_final")
        else:
            phase_timings = r.get("phase_timings") or {}
            n_extensions = sum(1 for k in phase_timings if k.startswith("conc_array_ext"))
            L_max = r.get("L_max")
            # A run that stopped short of the cap only does so because the
            # extension loop found the plume fit (see ATSimulation.run) —
            # that's a provably clean label. Hitting the cap is ambiguous
            # (the loop never re-checks after its last extension), so it's
            # excluded rather than guessed at.
            censored = (L_max is None) or (n_extensions >= max_extensions_at_run_time)
            dom_xmax_initial = config.dom_xmax
            dom_xmax_final = None

        rows.append({
            "config_name": r["config"],
            "config_hash": config_hash,
            "source_batch": os.path.basename(os.path.normpath(batch_dir)),
            **features,
            "L_max": r.get("L_max"),
            "dom_xmax_initial": dom_xmax_initial,
            "dom_xmax_final": dom_xmax_final,
            "n_extensions": n_extensions,
            "censored": censored,
            "wall_time_s": r.get("wall_time_s"),
        })

    return rows, n_skipped


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "batch_dirs", nargs="*",
        help="Directories with batch_configs/ + batch_results/test_results.json "
             "(pre-rename configs_test_batch/ + results_test_batch/ also accepted). "
             "Omit to just refit from the existing shared dataset.",
    )
    parser.add_argument("--dataset-path", default=DEFAULT_DATASET_PATH,
                         help=f"Shared dataset JSONL (default: {DEFAULT_DATASET_PATH})")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH,
                         help=f"Where to persist the refit model (default: {DEFAULT_MODEL_PATH})")
    parser.add_argument("--max-extensions-at-run-time", type=int, default=LEGACY_MAX_EXTENSIONS,
                         help=f"MAX_EXTENSIONS in effect when the batch(es) being ingested were "
                              f"run (default: {LEGACY_MAX_EXTENSIONS})")
    parser.add_argument("--quantile", type=float, default=0.90)
    parser.add_argument("--margin", type=float, default=1.15)
    parser.add_argument("--no-refit", action="store_true",
                         help="Only ingest into the dataset; don't retrain the model.")
    return parser.parse_args()


def main():
    args = _parse_args()

    existing_hashes = {row.get("config_hash") for row in load_dataset_rows(args.dataset_path)}
    new_rows = []
    for batch_dir in args.batch_dirs:
        try:
            rows, n_skipped = ingest_batch_dir(batch_dir, args.max_extensions_at_run_time)
        except FileNotFoundError as e:
            print(f"  {batch_dir}: {e}", file=sys.stderr)
            continue

        added = [r for r in rows if r["config_hash"] not in existing_hashes]
        for r in added:
            existing_hashes.add(r["config_hash"])
        new_rows.extend(added)

        n_censored = sum(1 for r in rows if r["censored"])
        print(f"{batch_dir}: {len(rows)} DONE rows ({n_censored} censored), "
              f"{len(added)} new, {n_skipped} skipped (missing config file)")

    if new_rows:
        os.makedirs(os.path.dirname(args.dataset_path), exist_ok=True)
        with open(args.dataset_path, "a") as f:
            for row in new_rows:
                f.write(json.dumps(row) + "\n")
        print(f"Appended {len(new_rows)} rows -> {args.dataset_path}")
    elif args.batch_dirs:
        print("No new rows to add (all configs already present in the dataset).")

    if args.no_refit:
        return

    predictor = XmaxPredictor(quantile=args.quantile, margin=args.margin,
                               model_path=args.model_path)
    n_clean = predictor.fit_from_dataset(args.dataset_path)
    predictor.save()
    total_rows = len(load_dataset_rows(args.dataset_path))
    print(f"Refit predictor on {n_clean} clean rows (of {total_rows} total) -> {args.model_path}")


if __name__ == "__main__":
    main()
