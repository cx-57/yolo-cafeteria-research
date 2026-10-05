import yaml

class DisposalEngine:
    def __init__(self, rules_path):
        with open(rules_path, "r") as file:
            self.rules = yaml.safe_load(file)

        self.fallback = self.rules.get("fallback", "manual_check")

    def classify(self, item_class):
        result = self.rules.get(item_class, self.fallback)

        if result == "TBD" or result is None:
            return self.fallback

        return result
