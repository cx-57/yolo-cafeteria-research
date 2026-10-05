"""Metrics for correct disposal guidance."""

from dataclasses import dataclass

@dataclass(frozen=True)
class DisposalResult:
    total_items: int
    correct: int
    abstained: int

    @property
    def coverage(self) -> float:
        return 0.0 if self.total_items == 0 else (self.total_items - self.abstained) / self.total_items

    @property
    def overall_accuracy(self) -> float:
        return 0.0 if self.total_items == 0 else self.correct / self.total_items

    @property
    def answered_accuracy(self) -> float:
        answered = self.total_items - self.abstained
        return 0.0 if answered == 0 else self.correct / answered
