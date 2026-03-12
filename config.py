"""
Hallucination Probe — Experiment Configuration
================================================
Centralised config for the observer model, device, and experiment hyper-parameters.
Change OBSERVER_MODEL here (or via CLI) to swap the frozen observer.
"""

import torch
from pathlib import Path

# ── Observer model ────────────────────────────────────────────
OBSERVER_MODEL: str = "meta-llama/Llama-3.1-8B"

# ── Device / dtype ────────────────────────────────────────────
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE: torch.dtype = torch.bfloat16 if DEVICE == "cuda" else torch.float32

# ── Paths ─────────────────────────────────────────────────────
PROJECT_ROOT: Path = Path(__file__).resolve().parent
DATA_PATH: Path = PROJECT_ROOT / "output" / "final_results.jsonl"
OUTPUT_DIR: Path = PROJECT_ROOT / "output"

# ── Experiment hyper-parameters ───────────────────────────────
N_FOLDS: int = 5
SEED: int = 42
MAX_SEQ_LEN: int = 4096
REGULARISATION_C_VALUES: list[float] = [1, 1e-1, 1e-2, 1e-3, 1e-4]  # grid of inverse regularisation strengths
