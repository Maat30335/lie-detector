#!/usr/bin/env python3
"""
Residual-Stream Linear Probe for Hallucination Detection
==========================================================
Implements the methodology from the paper excerpt:

1. Load evaluated summaries (article + LLM summary + faithful/hallucinated label).
2. Feed each (article, summary) pair through a frozen observer transformer
   via TransformerLens and cache residual-stream activations.
3. Extract the activation at the **final token** from every layer.
4. For each layer, train a logistic-regression probe with stratified
   5-fold cross-validation and report Accuracy, F1, and AUROC.
5. Select the best layer (highest mean AUROC) and save results.

Usage:
    python hallucination_probe.py                          # defaults from config.py
    python hallucination_probe.py --max-seq-len 128        # quick test
    python hallucination_probe.py --model meta-llama/Llama-3.1-8B-Instruct
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

import transformer_lens

import config as cfg


# ──────────────────────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────────────────────
def load_dataset(data_path: Path) -> tuple[list[str], np.ndarray]:
    """
    Load final_results.jsonl and return (texts, labels).

    Each text is the concatenation of the source article and the
    generated summary.  Labels: 0 = faithful, 1 = hallucinated.
    """
    texts: list[str] = []
    labels: list[int] = []

    with open(data_path, "r") as f:
        for line in f:
            rec = json.loads(line)
            # Skip examples with no conclusion
            if rec.get("conclusion") is None:
                continue
            # Concatenate article + summary as the observer input
            text = "[ARTICLE] " + rec["article"].strip() + " [SUMMARY]" + rec["llama_summary"].strip()
            texts.append(text)
            # conclusion == True means faithful (0), False means hallucinated (1)
            labels.append(0 if rec["conclusion"] is True else 1)

    labels_arr = np.array(labels, dtype=np.int64)
    print(f"[data] Loaded {len(texts)} examples  "
          f"(faithful={int((labels_arr == 0).sum())}, "
          f"hallucinated={int((labels_arr == 1).sum())})")
    return texts, labels_arr


# ──────────────────────────────────────────────────────────────
# Activation extraction
# ──────────────────────────────────────────────────────────────
def extract_activations(
    model: transformer_lens.HookedTransformer,
    texts: list[str],
    max_seq_len: int,
    device: str,
) -> torch.Tensor:
    """
    Run each text through the frozen observer and collect the
    post-layer-norm residual-stream vector at the final token
    from every layer.

    Returns:
        activations: Tensor of shape (num_examples, num_layers, d_model)
    """
    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model
    n_examples = len(texts)

    # Pre-allocate on CPU to avoid OOM on GPU
    all_activations = torch.zeros(
        (n_examples, n_layers, d_model), dtype=torch.float32
    )

    print(f"[activations] Extracting from {n_layers} layers, d_model={d_model}")

    for idx, text in enumerate(tqdm(texts, desc="Extracting activations")):
        # Tokenise with truncation
        tokens = model.to_tokens(text, prepend_bos=True)
        if tokens.shape[1] > max_seq_len:
            tokens = tokens[:, :max_seq_len]

        tokens = tokens.to(device)
        seq_len = tokens.shape[1]

        # Forward pass — cache all residual stream activations
        with torch.no_grad():
            _, cache = model.run_with_cache(
                tokens,
                names_filter=lambda name: name.endswith("hook_resid_post"),
                remove_batch_dim=True
            )

        # Extract the activation at the final token from each layer
        for layer in range(n_layers):
            hook_name = f"blocks.{layer}.hook_resid_post"
            # cache[hook_name] shape: (1, seq_len, d_model)
            act = cache[hook_name][seq_len - 1, :]  # final token
            all_activations[idx, layer, :] = act.float().cpu()

        # Free cache memory
        del cache

    return all_activations


# ──────────────────────────────────────────────────────────────
# Probe training & evaluation
# ──────────────────────────────────────────────────────────────
def run_layer_sweep(
    activations: torch.Tensor,
    labels: np.ndarray,
    n_folds: int,
    seed: int,
    reg_C: float,
) -> list[dict]:
    """
    For each layer, run stratified k-fold CV with logistic regression.

    Returns a list of dicts, one per layer, containing per-fold and
    mean metrics.
    """
    n_layers = activations.shape[1]
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    results: list[dict] = []

    for layer in range(n_layers):
        X = activations[:, layer, :].numpy()  # (N, D)
        fold_metrics: list[dict] = []

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, labels)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = labels[train_idx], labels[test_idx]

            clf = LogisticRegression(
                C=reg_C,
                max_iter=1000,
                solver="lbfgs",
                random_state=seed,
            )
            clf.fit(X_train, y_train)

            y_pred = clf.predict(X_test)
            y_prob = clf.predict_proba(X_test)

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, zero_division=0.0)

            # AUROC — handle edge cases where only one class is in the fold
            try:
                auroc = roc_auc_score(y_test, y_prob[:, 1])
            except ValueError:
                auroc = float("nan")

            fold_metrics.append({"acc": acc, "f1": f1, "auroc": auroc})

        # Aggregate across folds
        accs = [m["acc"] for m in fold_metrics]
        f1s = [m["f1"] for m in fold_metrics]
        aurocs = [m["auroc"] for m in fold_metrics if not np.isnan(m["auroc"])]

        layer_result = {
            "layer": layer,
            "acc_mean": float(np.mean(accs)),
            "acc_std": float(np.std(accs)),
            "f1_mean": float(np.mean(f1s)),
            "f1_std": float(np.std(f1s)),
            "auroc_mean": float(np.mean(aurocs)) if aurocs else float("nan"),
            "auroc_std": float(np.std(aurocs)) if aurocs else float("nan"),
            "folds": fold_metrics,
        }
        results.append(layer_result)

    return results


def select_best_layer(results: list[dict]) -> int:
    """Return the layer index with the highest mean AUROC."""
    valid = [(r["layer"], r["auroc_mean"]) for r in results if not np.isnan(r["auroc_mean"])]
    if not valid:
        print("[warning] No valid AUROC values — falling back to accuracy")
        return max(results, key=lambda r: r["acc_mean"])["layer"]
    return max(valid, key=lambda x: x[1])[0]


# ──────────────────────────────────────────────────────────────
# Reporting
# ──────────────────────────────────────────────────────────────
def print_results_table(results: list[dict], best_layer: int):
    """Pretty-print per-layer metrics."""
    header = f"{'Layer':>6}  {'Acc':>12}  {'F1':>12}  {'AUROC':>12}"
    print("\n" + "=" * 52)
    print("LAYER SWEEP RESULTS")
    print("=" * 52)
    print(header)
    print("-" * 52)

    for r in results:
        marker = " ★" if r["layer"] == best_layer else ""
        auroc_str = (
            f"{r['auroc_mean']:.3f}±{r['auroc_std']:.3f}"
            if not np.isnan(r["auroc_mean"])
            else "  N/A"
        )
        print(
            f"{r['layer']:>6}  "
            f"{r['acc_mean']:.3f}±{r['acc_std']:.3f}  "
            f"{r['f1_mean']:.3f}±{r['f1_std']:.3f}  "
            f"{auroc_str}{marker}"
        )

    print("=" * 52)
    print(f"Best layer: {best_layer} (highest mean AUROC)")


def save_results(
    results: list[dict],
    best_layer: int,
    model_name: str,
    output_dir: Path,
):
    """Save full results to JSON."""
    out_path = output_dir / "probe_results.json"
    payload = {
        "observer_model": model_name,
        "best_layer": best_layer,
        "per_layer": results,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults saved to {out_path}")


def save_best_probe(
    activations: torch.Tensor,
    labels: np.ndarray,
    best_layer: int,
    reg_C: float,
    seed: int,
    output_dir: Path,
):
    """
    Re-train on the full dataset at the best layer and save
    the probe weights and bias for later use.
    """
    X = activations[:, best_layer, :].numpy()
    clf = LogisticRegression(C=reg_C, max_iter=1000, solver="lbfgs", random_state=seed)
    clf.fit(X, labels)

    probe_path = output_dir / "probe_weights.npz"
    np.savez(
        probe_path,
        weights=clf.coef_[0],
        bias=clf.intercept_[0],
        best_layer=best_layer,
    )
    print(f"Probe weights saved to {probe_path}")


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Train a residual-stream linear probe for hallucination detection."
    )
    parser.add_argument(
        "--model", type=str, default=cfg.OBSERVER_MODEL,
        help=f"Observer model name for TransformerLens (default: {cfg.OBSERVER_MODEL})",
    )
    parser.add_argument(
        "--data-path", type=str, default=str(cfg.DATA_PATH),
        help=f"Path to final_results.jsonl (default: {cfg.DATA_PATH})",
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(cfg.OUTPUT_DIR),
        help=f"Directory for output files (default: {cfg.OUTPUT_DIR})",
    )
    parser.add_argument(
        "--max-seq-len", type=int, default=cfg.MAX_SEQ_LEN,
        help=f"Max token sequence length (default: {cfg.MAX_SEQ_LEN})",
    )
    parser.add_argument(
        "--device", type=str, default=cfg.DEVICE,
        help=f"Device to run on (default: {cfg.DEVICE})",
    )
    parser.add_argument(
        "--seed", type=int, default=cfg.SEED,
        help=f"Random seed (default: {cfg.SEED})",
    )
    parser.add_argument(
        "--n-folds", type=int, default=cfg.N_FOLDS,
        help=f"Number of CV folds (default: {cfg.N_FOLDS})",
    )
    parser.add_argument(
        "--reg-C", type=float, default=cfg.REGULARISATION_C,
        help=f"Logistic regression regularisation C (default: {cfg.REGULARISATION_C})",
    )
    args = parser.parse_args()

    data_path = Path(args.data_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Observer model : {args.model}")
    print(f"Device         : {args.device}")
    print(f"Max seq len    : {args.max_seq_len}")
    print(f"CV folds       : {args.n_folds}")
    print(f"Seed           : {args.seed}")
    print()

    t0 = time.time()

    # ── 1. Load data ──────────────────────────────────────────
    texts, labels = load_dataset(data_path)
    if len(texts) == 0:
        print("[error] No valid examples found. Exiting.")
        sys.exit(1)

    # ── 2. Load observer model ────────────────────────────────
    print(f"\n[model] Loading {args.model} via TransformerLens …")
    model = transformer_lens.HookedTransformer.from_pretrained(
        args.model,
        device=args.device,
        dtype=cfg.DTYPE,
    )
    model.eval()
    print(f"[model] Loaded — {model.cfg.n_layers} layers, d_model={model.cfg.d_model}")

    # ── 3. Extract activations ────────────────────────────────
    activations = extract_activations(model, texts, args.max_seq_len, args.device)
    print(f"[activations] Shape: {activations.shape}")

    # Free model from GPU after extraction
    del model
    if args.device == "cuda":
        torch.cuda.empty_cache()

    # ── 4. Layer sweep ────────────────────────────────────────
    print(f"\n[probe] Running {args.n_folds}-fold CV across all layers …")
    results = run_layer_sweep(activations, labels, args.n_folds, args.seed, args.reg_C)
    best_layer = select_best_layer(results)

    # ── 5. Report ─────────────────────────────────────────────
    print_results_table(results, best_layer)
    save_results(results, best_layer, args.model, output_dir)
    save_best_probe(activations, labels, best_layer, args.reg_C, args.seed, output_dir)

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed / 60:.1f} minutes")


if __name__ == "__main__":
    main()
