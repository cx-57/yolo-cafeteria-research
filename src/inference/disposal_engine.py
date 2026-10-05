"""Map detected physical item classes to locally verified disposal guidance."""

from pathlib import Path
import yaml

class DisposalEngine:
    def __init__(self, rules_path: str | Path):
        rules = yaml.safe_load(Path(rules_path).read_text())
        self.fallback = rules.pop("fallback", "manual_check")
        self.rules = rules

    def classify(self, item_class: str) -> str:
        stream = self.rules.get(item_class, self.fallback)
        if stream in (None, "TBD"):
            return self.fallback
        return str(stream)
