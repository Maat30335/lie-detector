#!/usr/bin/env python3
"""
Lookback Lens for Hallucination Detection
==========================================
Implements the Lookback Lens method from the paper:

1. Load evaluated summaries (article + LLM summary + faithful/hallucinated label).
2. Feed each (article, summary) pair through a frozen observer transformer
   via TransformerLens, caching attention patterns from all layers.
3. Compute lookback ratios — the ratio of attention on context (article)
   tokens vs. newly generated (summary) tokens — for every layer and head.
4. Average lookback ratios across summary tokens and concatenate across
   all layers/heads into a single feature vector per example.
5. Train a logistic-regression classifier with stratified k-fold CV
   and report Accuracy, F1, and AUROC.

Usage:
    python lookback_lens.py                          # defaults from config.py
    python lookback_lens.py --max-seq-len 128        # quick test
    python lookback_lens.py --model meta-llama/Llama-3.1-8B-Instruct
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
def load_dataset(
    data_path: Path,
) -> tuple[list[str], list[str], np.ndarray]:
    """
    Load final_results.jsonl and return (articles, summaries, labels).

    Keeps article and summary text separate so they can be tokenised
    independently for lookback-ratio computation.

    Labels: 0 = faithful, 1 = hallucinated.
    """
    articles: list[str] = []
    summaries: list[str] = []
    labels: list[int] = []

    with open(data_path, "r") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("conclusion") is None:
                continue
            articles.append(rec["article"].strip())
            summaries.append(rec["llama_summary"].strip())
            labels.append(0 if rec["conclusion"] is True else 1)

    labels_arr = np.array(labels, dtype=np.int64)
    print(
        f"[data] Loaded {len(articles)} examples  "
        f"(faithful={int((labels_arr == 0).sum())}, "
        f"hallucinated={int((labels_arr == 1).sum())})"
    )
    return articles, summaries, labels_arr


# ──────────────────────────────────────────────────────────────
# Lookback-ratio extraction
# ──────────────────────────────────────────────────────────────
def extract_lookback_ratios(
    model: transformer_lens.HookedTransformer,
    articles: list[str],
    summaries: list[str],
    max_seq_len: int,
    device: str,
) -> np.ndarray:
    """
    For each (article, summary) pair, compute the lookback ratio
    feature vector by:

    1. Tokenise article and summary separately, concatenate tokens,
       and record the boundary (n_context = len(article_tokens)).
    2. Run the concatenated sequence through the model with hooks
       on each layer's attention pattern (no caching).
    3. Each hook computes, for every summary token position t and head h:
         A_context = mean(attn[t, :n_context])
         A_new     = mean(attn[t, n_context:t])   (preceding summary tokens)
         LR        = A_context / (A_context + A_new)
       and accumulates the result into a shared tensor.
    4. Average LR across summary token positions → shape (L*H,).

    Returns:
        features: ndarray of shape (num_examples, n_layers * n_heads)
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    n_examples = len(articles)
    feature_dim = n_layers * n_heads

    all_features = np.zeros((n_examples, feature_dim), dtype=np.float32)

    print(
        f"[lookback] Extracting ratios from {n_layers} layers × "
        f"{n_heads} heads = {feature_dim} features"
    )

    for idx in tqdm(range(n_examples), desc="Extracting lookback ratios"):
        # ── Tokenise article and summary separately ──────────
        article_tokens = model.to_tokens(articles[idx], prepend_bos=True)  # (1, N_a)
        summary_tokens = model.to_tokens(summaries[idx], prepend_bos=False)  # (1, N_s)

        # Concatenate: [BOS, article_tokens..., summary_tokens...]
        tokens = torch.cat([article_tokens, summary_tokens], dim=1)  # (1, N)

        n_context = article_tokens.shape[1]  # includes BOS

        # Truncate if necessary
        if tokens.shape[1] > max_seq_len:
            tokens = tokens[:, :max_seq_len]

        tokens = tokens.to(device)
        seq_len = tokens.shape[1]

        # If truncation removed all summary tokens, skip
        if n_context >= seq_len:
            continue

        # ── Prepare accumulator for lookback ratios ──────────
        # Each hook writes to its own layer row; no post-hoc
        # iteration over the cache is needed.
        layer_head_ratios = torch.zeros(
            (n_layers, n_heads), dtype=torch.float32
        )
        n_summary_tokens = seq_len - n_context

        def make_lookback_hook(layer_idx, n_ctx, sl, n_h):
            """Return a hook fn that computes lookback ratios for one layer."""
            def hook_fn(attn_pattern, hook):
                # attn_pattern shape: (batch, n_heads, seq_len, seq_len)
                attn = attn_pattern[0]  # (n_heads, seq_len, seq_len)
                for t in range(n_ctx, sl):
                    a_context = attn[:, t, :n_ctx].mean(dim=1)  # (n_heads,)
                    if t > n_ctx:
                        a_new = attn[:, t, n_ctx:t].mean(dim=1)  # (n_heads,)
                    else:
                        a_new = torch.zeros(n_h, device=attn.device)
                    denom = a_context + a_new
                    lr = torch.where(
                        denom > 0,
                        a_context / denom,
                        torch.ones_like(denom) * 0.5,
                    )
                    layer_head_ratios[layer_idx] += lr.cpu()
                return attn_pattern
            return hook_fn

        # Build list of (hook_name, hook_fn) pairs
        hook_fns = [
            (
                f"blocks.{layer}.attn.hook_pattern",
                make_lookback_hook(layer, n_context, seq_len, n_heads),
            )
            for layer in range(n_layers)
        ]

        # ── Forward pass — hooks compute ratios on the fly ───
        with torch.no_grad():
            model.run_with_hooks(
                tokens,
                fwd_hooks=hook_fns,
            )

        # Average across summary tokens
        layer_head_ratios /= n_summary_tokens

        # Flatten to (L*H,) and store
        all_features[idx] = layer_head_ratios.flatten().numpy()

    return all_features


# ──────────────────────────────────────────────────────────────
# Probe training & evaluation  (all-layers)
# ──────────────────────────────────────────────────────────────
def run_all_layers_cv(
    features: np.ndarray,
    labels: np.ndarray,
    n_folds: int,
    seed: int,
    reg_C_values: list[float],
) -> dict:
    """
    Train a logistic-regression probe on the full L*H lookback-ratio
    feature vector with stratified k-fold CV, grid-searching over
    regularisation constants.

    Returns a dict containing the best C, per-fold metrics, mean
    metrics, and a summary of all C values tried.
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    best_f1 = -1.0
    best_c_result: dict | None = None
    all_c_results: list[dict] = []

    for C_val in reg_C_values:
        fold_metrics: list[dict] = []

        for fold_idx, (train_idx, test_idx) in enumerate(
            skf.split(features, labels)
        ):
            X_train, X_test = features[train_idx], features[test_idx]
            y_train, y_test = labels[train_idx], labels[test_idx]

            clf = LogisticRegression(
                C=C_val,
                max_iter=1000,
                solver="lbfgs",
                random_state=seed,
            )
            clf.fit(X_train, y_train)

            y_pred = clf.predict(X_test)
            y_prob = clf.predict_proba(X_test)

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, zero_division=0.0)

            try:
                auroc = roc_auc_score(y_test, y_prob[:, 1])
            except ValueError:
                auroc = float("nan")

            fold_metrics.append({"acc": acc, "f1": f1, "auroc": auroc})

        # Aggregate across folds
        accs = [m["acc"] for m in fold_metrics]
        f1s = [m["f1"] for m in fold_metrics]
        aurocs = [
            m["auroc"] for m in fold_metrics if not np.isnan(m["auroc"])
        ]

        c_result = {
            "C": C_val,
            "acc_mean": float(np.mean(accs)),
            "acc_std": float(np.std(accs)),
            "f1_mean": float(np.mean(f1s)),
            "f1_std": float(np.std(f1s)),
            "auroc_mean": float(np.mean(aurocs)) if aurocs else float("nan"),
            "auroc_std": float(np.std(aurocs)) if aurocs else float("nan"),
            "folds": fold_metrics,
        }
        all_c_results.append(c_result)

        mean_f1 = c_result["f1_mean"]
        print(f"  C={C_val:<8g}  F1={mean_f1:.4f}")

        if mean_f1 > best_f1:
            best_f1 = mean_f1
            best_c_result = c_result

    # Fall back to accuracy if all F1s are zero
    if best_c_result is None:
        best_c_result = max(all_c_results, key=lambda r: r["acc_mean"])

    return {
        "best_C": best_c_result["C"],
        "acc_mean": best_c_result["acc_mean"],
        "acc_std": best_c_result["acc_std"],
        "f1_mean": best_c_result["f1_mean"],
        "f1_std": best_c_result["f1_std"],
        "auroc_mean": best_c_result["auroc_mean"],
        "auroc_std": best_c_result["auroc_std"],
        "folds": best_c_result["folds"],
        "all_C_results": all_c_results,
    }


# ──────────────────────────────────────────────────────────────
# Reporting
# ──────────────────────────────────────────────────────────────
def print_results(result: dict):
    """Pretty-print all-layers lookback lens results."""
    print("\n" + "=" * 60)
    print("LOOKBACK LENS RESULTS  (all layers)")
    print("=" * 60)

    auroc_str = (
        f"{result['auroc_mean']:.3f}±{result['auroc_std']:.3f}"
        if not np.isnan(result["auroc_mean"])
        else "  N/A"
    )
    print(f"  Best C   : {result['best_C']:g}")
    print(f"  Accuracy : {result['acc_mean']:.3f}±{result['acc_std']:.3f}")
    print(f"  F1       : {result['f1_mean']:.3f}±{result['f1_std']:.3f}")
    print(f"  AUROC    : {auroc_str}")
    print("=" * 60)


def save_results(
    result: dict,
    model_name: str,
    output_dir: Path,
):
    """Save full results to JSON."""
    out_path = output_dir / "lookback_lens_results.json"
    payload = {
        "observer_model": model_name,
        "method": "lookback_lens_all_layers",
        "result": result,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults saved to {out_path}")


def save_best_probe(
    features: np.ndarray,
    labels: np.ndarray,
    best_C: float,
    seed: int,
    output_dir: Path,
):
    """
    Re-train on the full dataset with the best C and save the
    probe weights and bias for later use.
    """
    clf = LogisticRegression(
        C=best_C, max_iter=1000, solver="lbfgs", random_state=seed
    )
    clf.fit(features, labels)

    probe_path = output_dir / "lookback_lens_weights.npz"
    np.savez(
        probe_path,
        weights=clf.coef_[0],
        bias=clf.intercept_[0],
        best_C=best_C,
    )
    print(f"Probe weights saved to {probe_path}")


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Lookback Lens for hallucination detection."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=cfg.OBSERVER_MODEL,
        help=f"Observer model name for TransformerLens (default: {cfg.OBSERVER_MODEL})",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=str(cfg.DATA_PATH),
        help=f"Path to final_results.jsonl (default: {cfg.DATA_PATH})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(cfg.OUTPUT_DIR),
        help=f"Directory for output files (default: {cfg.OUTPUT_DIR})",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=cfg.MAX_SEQ_LEN,
        help=f"Max token sequence length (default: {cfg.MAX_SEQ_LEN})",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=cfg.DEVICE,
        help=f"Device to run on (default: {cfg.DEVICE})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=cfg.SEED,
        help=f"Random seed (default: {cfg.SEED})",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=cfg.N_FOLDS,
        help=f"Number of CV folds (default: {cfg.N_FOLDS})",
    )
    parser.add_argument(
        "--reg-C-values",
        type=float,
        nargs="+",
        default=cfg.REGULARISATION_C_VALUES,
        help=f"Regularisation C values to grid-search (default: {cfg.REGULARISATION_C_VALUES})",
    )
    args = parser.parse_args()

    data_path = Path(args.data_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Observer model : {args.model}")
    print(f"Device         : {args.device}")
    print(f"Max seq len    : {args.max_seq_len}")
    print(f"CV folds       : {args.n_folds}")
    print(f"Reg C values   : {args.reg_C_values}")
    print(f"Seed           : {args.seed}")
    print()

    t0 = time.time()

    # ── 1. Load data ──────────────────────────────────────────
    articles, summaries, labels = load_dataset(data_path)
    if len(articles) == 0:
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
    print(
        f"[model] Loaded — {model.cfg.n_layers} layers, "
        f"{model.cfg.n_heads} heads, d_model={model.cfg.d_model}"
    )

    # ── 3. Extract lookback ratios ────────────────────────────
    features = extract_lookback_ratios(
        model, articles, summaries, args.max_seq_len, args.device
    )
    print(f"[lookback] Feature matrix shape: {features.shape}")

    # Free model from GPU after extraction
    del model
    if args.device == "cuda":
        torch.cuda.empty_cache()

    # ── 4. Train probe (all layers) ───────────────────────────
    print(
        f"\n[probe] Running {args.n_folds}-fold CV × "
        f"{len(args.reg_C_values)} C values …"
    )
    result = run_all_layers_cv(
        features, labels, args.n_folds, args.seed, args.reg_C_values
    )

    # ── 5. Report ─────────────────────────────────────────────
    print_results(result)
    save_results(result, args.model, output_dir)
    save_best_probe(
        features, labels, result["best_C"], args.seed, output_dir
    )

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed / 60:.1f} minutes")


if __name__ == "__main__":
    main()
