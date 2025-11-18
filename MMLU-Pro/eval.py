import json
import argparse
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate MMLU-Pro predictions against ground truth answers.",
    )
    parser.add_argument(
        "--file",
        type=Path,
        required=True,
        help="Path to the JSONL file containing model predictions.",
    )
    args = parser.parse_args()

    preds = []
    answers = []

    with args.file.open("r", encoding="utf-8") as f:
        for line in map(json.loads, f):
            # Parse ground-truth answers
            answer = line["outputs"]

            # Extract text inside The answer is (...)
            pred = line["pred"].split("The answer is (")[-1].split(")")[0].strip()

            answers.append(answer)
            preds.append(pred)

    # Count predictions that exactly match one of the valid answers
    correct = sum(p in a for p, a in zip(preds, answers))
    total = len(answers)
    accuracy = correct / total * 100.0
    print(f"Accuracy: {accuracy:.2f}% ({correct}/{total})")
