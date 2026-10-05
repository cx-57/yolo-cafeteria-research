def calculate_disposal_metrics(total_items, correct, abstained):
    if total_items == 0:
        return {
            "overall_accuracy": 0,
            "coverage": 0,
            "answered_accuracy": 0,
        }

    answered = total_items - abstained

    overall_accuracy = correct / total_items
    coverage = answered / total_items

    if answered == 0:
        answered_accuracy = 0
    else:
        answered_accuracy = correct / answered

    return {
        "overall_accuracy": overall_accuracy,
        "coverage": coverage,
        "answered_accuracy": answered_accuracy,
    }
