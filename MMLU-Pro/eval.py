import json
import argparse
from pathlib import Path

list_of_answers = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

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
    out_of_lines = []

    with args.file.open("r", encoding="utf-8") as f:
        for line in map(json.loads, f):
            # Parse ground-truth answers
            answer = line["outputs"]
            id = line["index"]

            # Extract text inside The answer is (...)
            pred = line["pred"].split("answer is ")[-1].strip()
            if "(" in pred and ")" in pred:
                pred = pred.split("(")[1].split(")")[0].strip()
            else:
                pred = pred[0].strip()  # Fallback: take the first character
            if pred not in list_of_answers:
                out_of_lines.append(line)

            answers.append(answer)
            preds.append(pred)

    # Count predictions that exactly match one of the valid answers
    correct = sum(p in a for p, a in zip(preds, answers))
    total = len(answers)
    accuracy = correct / total * 100.0
    print(f"Accuracy: {accuracy:.2f}% ({correct}/{total})")
    out_of_line_file = args.file.with_name(args.file.stem + "_out_of_lines.jsonl")

    with out_of_line_file.open("w", encoding="utf-8") as f:
        for item in out_of_lines:
            f.write(json.dumps(item) + "\n")
