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
    wrong_by_out_of_line = 0

    with args.file.open("r", encoding="utf-8") as f:
        for line in map(json.loads, f):
            # Parse ground-truth answers
            answer = [float(obj) for obj in line["outputs"]]
            id = line["index"]

            # Extract text inside \boxed{}
            pred_raw = line["pred"].split("\\boxed{")[-1].split("}")[0]
            pred_raw = pred_raw.replace(",", "").strip()

            # Try converting prediction to float; fallback to +inf
            if "$" in pred_raw:
                pred_raw = pred_raw.replace("$", "")
            if " " in pred_raw:
                pred_raw = pred_raw.replace(" ", "")
            try:
                pred = float(pred_raw)
            except ValueError:
                    wrong_by_out_of_line += 1
                    pred = float("inf")

            answers.append(answer)
            preds.append(pred)

    # Count predictions that exactly match one of the valid answers
    correct = sum(p in a for p, a in zip(preds, answers))
    total = len(answers)
    accuracy = correct / total * 100.0
    print(f"Accuracy: {accuracy:.2f}% ({correct}/{total})")
    out_of_line_rate = wrong_by_out_of_line / total * 100.0
    print(f"Out-of-line prediction rate: {out_of_line_rate:.2f}% ({wrong_by_out_of_line}/{total})")
