"""Transparent risk banding.

Thresholds are NOT hand-picked. `scripts/train_models.py` derives them from out-of-fold
predictions on the training split:

* HIGH   - probability >= the decision threshold, chosen to maximise F2 (recall weighted
           4x as heavily as precision, reflecting that a missed failure costs more than a
           false alarm).
* MEDIUM - probability >= the "watch" threshold: the highest threshold that still catches
           at least `watch_recall` (default 95%) of training failures out-of-fold.
* LOW    - everything else.

They are then checked on the validation split and reported on the untouched test split,
together with the observed failure rate inside each band. Both thresholds can be
overridden at runtime (env vars or the API's `thresholds` field) without retraining.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

LOW, MEDIUM, HIGH = "LOW", "MEDIUM", "HIGH"


@dataclass(frozen=True)
class RiskPolicy:
    medium_threshold: float
    high_threshold: float
    method: str = ""

    def __post_init__(self) -> None:
        if not 0.0 < self.medium_threshold <= self.high_threshold < 1.0:
            raise ValueError(
                "Thresholds must satisfy 0 < medium <= high < 1 "
                f"(got medium={self.medium_threshold}, high={self.high_threshold})"
            )

    def classify(self, probability: float) -> str:
        if probability >= self.high_threshold:
            return HIGH
        if probability >= self.medium_threshold:
            return MEDIUM
        return LOW

    def to_dict(self) -> dict:
        return asdict(self)
