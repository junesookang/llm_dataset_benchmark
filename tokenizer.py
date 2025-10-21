import argparse
import json
from pathlib import Path

from template import Templates, SYSTEM_PROMPT, BOS_TOKEN
from task_summary import TASKS


def resolve_template(model_name: str) -> tuple[str, str]:
    """Return the template string for the tokenizer builder with fallback."""
    builder = model_name.split("/")[0]
    template = Templates.get(builder, Templates["base"])
    sys_prompt = SYSTEM_PROMPT.get(builder, SYSTEM_PROMPT["base"])
    return template, builder, sys_prompt


def iter_jsonl(path: Path):
    """Yield (line_number, record) pairs from a JSONL file."""
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            yield line_number, json.loads(line)


def format_record(record: dict, template: str, sys_prompt: str) -> dict:
    prompt = template.format(system_prompt=sys_prompt, task_template=record["prompt"])
    outputs = record.get("outputs", []) + [record.get("answer")] if "answer" in record else []
    return {
        "index": record["index"],
        "prompt": prompt,
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

    if args.major_task == "mmlu-pro":
        template = Templates["base"]
        builder = args.model_name.split("/")[0]
        prompt_format = TASKS.get(args.major_task)["system_prompt"]
    else:
        template, builder, system_prompt = resolve_template(args.model_name)
    if builder not in Templates:
        print(f"Warning: no template registered for '{builder}'. Falling back to 'base'.")

    output_dir = Path(TASKS.get(args.major_task)["dataset_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = output_dir.parent / "datasets"
    files = TASKS.get(args.major_task)["files"]

    for file in files:
        input_file = dataset_dir / file
        output_path = output_dir / input_file.name

        task_name = file.split("-")[-1].replace(".jsonl", "")
        if args.major_task == "mmlu-pro":
            system_prompt = BOS_TOKEN.get(builder, "") + prompt_format.replace("{$}", task_name) + "\n\n"
        with output_path.open("w", encoding="utf-8") as handle:
            for _, record in iter_jsonl(input_file):
                formatted = format_record(record, template, system_prompt)
                handle.write(json.dumps(formatted, ensure_ascii=False) + "\n")

        print(f"Tokenized data saved to {output_path}")


if __name__ == "__main__":
    main()

