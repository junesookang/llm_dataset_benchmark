import argparse
import json
from pathlib import Path

from template import Templates, THINK_TOKEN, BOS_TOKEN, EOS_TOKEN
from task_summary import TASKS


def iter_jsonl(path: Path):
    """Yield (line_number, record) pairs from a JSONL file."""
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            yield line_number, json.loads(line)


def format_record(record: dict, builder: str, add_think_token: bool) -> dict:
    builder_template = Templates.get(builder)
    eos = EOS_TOKEN.get(builder)
    prompt = [BOS_TOKEN.get(builder)]
    lengths = len(record["prompt"])
    for i, p in enumerate(record["prompt"]):
        prompt.append(builder_template.format(task_template=p["user"]) + p["assistant"])
        if i < lengths - 1: # Don't add assistant eos for the last turn
            prompt.append(eos)
    if add_think_token:
        prompt.append(THINK_TOKEN)
    outputs = record.get("outputs", []) + [record.get("answer")] if "answer" in record else []
    return {
        "index": record["index"],
        "prompt": "".join(prompt),
        "outputs": outputs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply the chat template for a model and emit JSONL with formatted prompts.",
    )
    parser.add_argument(
        "--major-task",
        type=str,
        required=True,
        help="Major task identifier within the benchmark (e.g., mmlu-pro, gsm8k).",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Model name used to resolve the template (e.g., meta-llama/Llama-3.1-8B-Instruct).",
    )
    args = parser.parse_args()

    args.add_think_token = "Distill" in args.model_name.split("/")[-1]
    builder = args.model_name.split("/")[0]
    if builder not in Templates:
        print(f"Warning: no template registered for '{builder}'. Falling back to 'base'.")
        builder = "base"

    task_dir = Path(TASKS.get(args.major_task)["dataset_dir"])
    dataset_dir = task_dir / "datasets"
    files = TASKS.get(args.major_task)["files"]

    model_name = args.model_name.split("/")[-1]
    output_dir = Path(__file__).resolve().parent / "tokenized_datasets" / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    for file in files:
        input_file = dataset_dir / file
        output_path = output_dir / input_file.name

        with output_path.open("w", encoding="utf-8") as handle:
            for _, record in iter_jsonl(input_file):
                formatted = format_record(record, builder, args.add_think_token)
                handle.write(json.dumps(formatted, ensure_ascii=False) + "\n")

        print(f"Tokenized data saved to {output_path}")


if __name__ == "__main__":
    main()

