"""
calibrate_temperature.py - Fit a softmax temperature for the active stance checkpoint.

MC dropout (stance_detector.py) gives a *relative* uncertainty signal: it tells
you a prediction is shakier than another, but "confidence": 0.9 does not mean
"correct 90% of the time" -- neural nets are systematically overconfident.
Temperature scaling (Guo et al. 2017) fixes this cheaply: divide logits by a
single learned scalar T before softmax. It does not change which class wins
(T > 0 preserves argmax), only how honest the confidence number is. Unlike
train_stance.py, this needs no GPU -- it is a handful of forward passes over a
held-out set plus a 1-parameter optimization, so it runs on this laptop.

Usage:
    python -m backend.training.calibrate_temperature [--n-samples 1000]

Recalibrate whenever STANCE_CHECKPOINT changes -- T is specific to one model's
confidence/accuracy relationship, not a property of the task.
"""

import argparse
import logging

import numpy as np
import torch

from backend.config import config
from backend.models.stance_detector import StanceDetector, _LABEL_ALIASES
from backend.training.train_stance import load_fever, load_vitaminc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _load_eval_set(n_samples: int, seed: int = 42) -> list[dict]:
    """FEVER + VitaminC validation examples, in stance_detector's own schema."""
    _, fever_val = load_fever()
    _, vitaminc_val = load_vitaminc()
    from datasets import concatenate_datasets

    combined = concatenate_datasets([fever_val, vitaminc_val]).shuffle(seed=seed)
    n = min(n_samples, len(combined))
    return combined.select(range(n)).to_list()


def _collect_logits(detector: StanceDetector, examples: list[dict], batch_size: int = 32):
    """Single deterministic forward pass per example -- no MC dropout, no temperature."""
    canonical_to_index = {v: k for k, v in detector.label_map.items()}
    all_logits, all_targets = [], []

    for start in range(0, len(examples), batch_size):
        chunk = examples[start : start + batch_size]
        premises = [ex["premise"][: config.MAX_PREMISE_CHARS] for ex in chunk]
        hypotheses = [ex["hypothesis"] for ex in chunk]
        targets = []
        keep = []
        for i, ex in enumerate(chunk):
            label_str = {0: "REFUTES", 1: "NOT ENOUGH INFO", 2: "SUPPORTS"}[ex["label"]]
            canonical = _LABEL_ALIASES[label_str.lower()]
            if canonical not in canonical_to_index:
                continue  # checkpoint doesn't expose this class (shouldn't happen)
            targets.append(canonical_to_index[canonical])
            keep.append(i)

        inputs = detector.tokenizer(
            [premises[i] for i in keep],
            [hypotheses[i] for i in keep],
            truncation=True,
            max_length=512,
            padding=True,
            return_tensors="pt",
        ).to(detector.device)

        with torch.no_grad():
            logits = detector.model(**inputs).logits
        all_logits.append(logits.cpu())
        all_targets.extend(targets)

    return torch.cat(all_logits), torch.tensor(all_targets)


def _fit_temperature(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Minimize NLL(logits / T, targets) over a single scalar T, via LBFGS."""
    log_t = torch.zeros(1, requires_grad=True)  # optimize in log-space: T = exp(log_t) > 0 always
    optimizer = torch.optim.LBFGS([log_t], lr=0.05, max_iter=100)
    nll = torch.nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        loss = nll(logits / log_t.exp(), targets)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_t.exp().item())


def _expected_calibration_error(probs: np.ndarray, targets: np.ndarray, n_bins: int = 15) -> float:
    """Mean gap between confidence and accuracy within confidence bins -- lower is better."""
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = (predictions == targets).astype(float)

    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if not mask.any():
            continue
        bin_confidence = confidences[mask].mean()
        bin_accuracy = correct[mask].mean()
        ece += mask.mean() * abs(bin_confidence - bin_accuracy)
    return float(ece)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=1000)
    args = parser.parse_args()

    detector = StanceDetector()
    logger.info(
        "Calibrating checkpoint=%r (finetuned=%s)", detector.checkpoint, detector.is_finetuned
    )

    examples = _load_eval_set(args.n_samples)
    logger.info("Loaded %d held-out examples (FEVER + VitaminC validation)", len(examples))

    logits, targets = _collect_logits(detector, examples)
    probs_before = torch.softmax(logits, dim=-1).numpy()
    ece_before = _expected_calibration_error(probs_before, targets.numpy())

    temperature = _fit_temperature(logits, targets)

    probs_after = torch.softmax(logits / temperature, dim=-1).numpy()
    ece_after = _expected_calibration_error(probs_after, targets.numpy())

    accuracy = float((probs_before.argmax(axis=1) == targets.numpy()).mean())

    print(f"\nCheckpoint:        {detector.checkpoint}")
    print(f"Held-out examples: {len(examples)}")
    print(f"Accuracy:          {accuracy:.4f}  (temperature scaling does not change this)")
    print(f"ECE before:        {ece_before:.4f}")
    print(f"ECE after:         {ece_after:.4f}")
    print(f"\nTemperature = {temperature:.4f}")
    print(f"\nSet in .env: STANCE_TEMPERATURE={temperature:.4f}")


if __name__ == "__main__":
    main()
