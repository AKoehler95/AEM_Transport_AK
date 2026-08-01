# Written by Alvin Yadav

"""
Learned dom_xmax predictor.

Predicts an upper quantile of L_max (plume extent) from a config's physics +
source-geometry features, and converts it to an initial dom_xmax guess. The
dynamic domain-extension loop in ATSimulation.run() remains the correctness
guarantee — a bad prediction only costs an extra extension pass, never a
capped/incorrect result.

Model: sklearn GradientBoostingRegressor with a quantile loss. Learning is
append-and-refit (not per-sample SGD): update() buffers a labelled row,
refit() retrains on the full buffer. This is fast enough at the scale of a
few hundred/thousand rows to be called after every clean run.
"""

import json
import os

import joblib
import numpy as np

from xmax_features import feature_names

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Predictor artifacts live in their own directory, deliberately separate from
# the simulation output in sim_runs/ — they are model state, not run results.
MODEL_DIR = os.path.join(SCRIPT_DIR, "xmax_model")
DEFAULT_MODEL_PATH = os.path.join(MODEL_DIR, "xmax_model.pkl")
# Tracked in git (see .gitignore) so the dataset — not the pickled model,
# which is fragile across sklearn versions — is what carries learning
# between machines. Each machine refits its own local model from it.
DEFAULT_DATASET_PATH = os.path.join(MODEL_DIR, "lmax_dataset.jsonl")

MIN_TRAIN = 20          # need at least this many clean rows before the model is trusted
MIN_DOMAIN = 50.0       # never predict a domain smaller than this
LEGACY_MARGIN = 150.0   # matches generate_configs.base_config's fixed +150 rule
BOX_SLACK = 0.10        # relative slack allowed outside the observed training-feature range


def load_dataset_rows(path):
    """
    Read a dataset JSONL file, deduping by config_hash (last occurrence
    wins). Rows from multiple machines/batches can be concatenated freely —
    re-ingesting the same config never double-counts it.

    Rows without a config_hash (e.g. hand-written test fixtures) dedupe by
    config_name instead. Missing file returns an empty list.
    """
    if not os.path.exists(path):
        return []
    rows = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            key = row.get("config_hash") or row.get("config_name")
            rows[key] = row
    return list(rows.values())


class XmaxPredictor:
    def __init__(self, quantile=0.90, margin=1.15, model_path=None, min_train=MIN_TRAIN):
        self.quantile = quantile
        self.margin = margin
        self.min_train = min_train
        self.model_path = model_path or DEFAULT_MODEL_PATH

        self.model = None
        self.X = []   # buffered feature rows, in feature_names() order
        self.y = []   # buffered L_max labels
        self._feat_min = None
        self._feat_max = None

        self.load()

    def _vec(self, features: dict):
        names = feature_names()
        missing = [k for k in names if k not in features]
        if missing:
            raise ValueError(f"features dict missing keys: {missing}")
        return [features[k] for k in names]

    def _out_of_box(self, vec):
        if self._feat_min is None:
            return True
        for v, lo, hi in zip(vec, self._feat_min, self._feat_max):
            span = hi - lo
            slack = span * BOX_SLACK if span > 0 else abs(lo) * BOX_SLACK + 1e-9
            if v < lo - slack or v > hi + slack:
                return True
        return False

    def predict(self, features: dict) -> dict:
        """
        Returns {'L_pred_q', 'dom_xmax', 'trained', 'n_train'}.

        Falls back to the legacy '+150 beyond the furthest source point' rule
        whenever the model isn't trained yet or the config falls outside the
        feature ranges seen during training — this guarantees behaviour never
        regresses on unfamiliar inputs.
        """
        vec = self._vec(features)
        n_train = len(self.y)
        legacy_xmax = features["x_max_src"] + LEGACY_MARGIN

        usable = self.model is not None and n_train >= self.min_train and not self._out_of_box(vec)
        if not usable:
            return {
                "L_pred_q": None,
                "dom_xmax": max(legacy_xmax, MIN_DOMAIN),
                "trained": False,
                "n_train": n_train,
            }

        l_pred_q = float(self.model.predict(np.array([vec]))[0])
        dom_xmax = max(l_pred_q * self.margin, MIN_DOMAIN)
        return {
            "L_pred_q": l_pred_q,
            "dom_xmax": dom_xmax,
            "trained": True,
            "n_train": n_train,
        }

    def update(self, features: dict, L_max: float):
        """Buffer a clean (features, L_max) pair for the next refit()."""
        self.X.append(self._vec(features))
        self.y.append(float(L_max))

    def fit_from_dataset(self, dataset_path=None, replace=True):
        """
        Bulk-train directly from a dataset JSONL file, bypassing update().

        This is the entry point for learning from runs that didn't go
        through run_tests.py's live update()/refit() loop — historical
        batches backfilled by train_xmax_from_runs.py, or a dataset that
        just arrived via `git pull` from another machine. Rows are deduped
        by config_hash (see load_dataset_rows) and filtered to clean
        (non-censored, labelled) ones before fitting.

        replace=True discards the current in-memory buffer first, so the
        model reflects exactly what's in the file (the normal case for a
        shared, git-tracked dataset). Pass False to add to whatever was
        already loaded/update()-ed.

        Returns the number of clean rows the model was fit on.
        """
        dataset_path = dataset_path or DEFAULT_DATASET_PATH
        rows = load_dataset_rows(dataset_path)
        names = feature_names()

        if replace:
            self.X, self.y = [], []

        n_used = 0
        for row in rows:
            if row.get("censored") or row.get("L_max") is None:
                continue
            try:
                vec = [row[k] for k in names]
            except KeyError:
                # Row predates a feature-set change (missing keys) — skip
                # rather than crash; it'll disappear once regenerated.
                continue
            self.X.append(vec)
            self.y.append(float(row["L_max"]))
            n_used += 1

        self.refit()
        return n_used

    def refit(self):
        """Refit the quantile model on the full buffer, if there's enough data."""
        if len(self.y) < 2:
            return

        from sklearn.ensemble import GradientBoostingRegressor

        X_arr = np.array(self.X)
        y_arr = np.array(self.y)
        self._feat_min = X_arr.min(axis=0)
        self._feat_max = X_arr.max(axis=0)

        if len(y_arr) >= self.min_train:
            model = GradientBoostingRegressor(
                loss="quantile", alpha=self.quantile,
                n_estimators=150, max_depth=3, learning_rate=0.1,
                random_state=0,
            )
            model.fit(X_arr, y_arr)
            self.model = model

    def save(self):
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump({
            "model": self.model,
            "X": self.X,
            "y": self.y,
            "feat_min": self._feat_min,
            "feat_max": self._feat_max,
            "quantile": self.quantile,
            "margin": self.margin,
            "min_train": self.min_train,
        }, self.model_path)

    def load(self):
        if not os.path.exists(self.model_path):
            return
        data = joblib.load(self.model_path)
        self.model = data.get("model")
        self.X = data.get("X", [])
        self.y = data.get("y", [])
        self._feat_min = data.get("feat_min")
        self._feat_max = data.get("feat_max")


def suggest_dom_xmax(config, predictor: "XmaxPredictor | None" = None) -> float:
    """
    One-shot entry point for single-run callers (GUI / designer_solver adapter):
    extract features from `config`, load the persisted model, and return a
    suggested dom_xmax. Falls back to the legacy rule if no model is trained
    or the config is out of range.
    """
    from xmax_features import extract_features

    predictor = predictor or XmaxPredictor()
    features = extract_features(config)
    return predictor.predict(features)["dom_xmax"]
