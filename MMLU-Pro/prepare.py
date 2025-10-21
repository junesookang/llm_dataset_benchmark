import argparse
import json
import random
import re
from pathlib import Path

from datasets import load_dataset

choices = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
max_model_length = 4096
max_new_tokens = 2048
random.seed(12345)


def load_mmlu_pro():
    dataset = load_dataset("TIGER-Lab/MMLU-Pro")
    test_df, val_df = dataset["test"], dataset["validation"]
    test_df = preprocess(test_df)
    val_df = preprocess(val_df)
    return test_df, val_df


def preprocess(test_df):
    res_df = []
    for each in test_df:
        options = [opt for opt in each["options"] if opt != "N/A"]
        example = dict(each)
        example["options"] = options
        res_df.append(example)
    return res_df


def select_by_category(df, subject, exclude_question=None):
    res = []
    for each in df:
        if each["category"] != subject:
            continue
        if exclude_question is not None and each["question"] == exclude_question:
            continue
        res.append(each)
    return res


def format_cot_example(example, including_answer=True):
    prompt_parts = [
        f"Question: {example['question']}\n",
        "Options: ",
    ]
    prompt_parts.extend(
        f"{choices[i]}. {opt}\n" for i, opt in enumerate(example["options"])
    )

    if including_answer:
        cot_content = (example.get("cot_content") or "").strip()
        if cot_content == "":
            cot_content = "Let's think step by step."
        if cot_content.startswith("A: "):
            cot_content = cot_content[3:].strip()
    else:
        cot_content = "Let's think step by step."

    content = f"Answer: {cot_content}" + ("\n\n" if including_answer else "")
    prompt_parts.append(content)
    return "".join(prompt_parts)

def generate_cot_prompt(val_df, curr, k):
    subject = curr["category"]
    in_context_examples = select_by_category(val_df, subject, exclude_question=curr["question"])
    random.shuffle(in_context_examples)
    in_context_examples = in_context_examples[:k]
    prompt = ""
    for example in in_context_examples:
        prompt += format_cot_example(example, including_answer=True)
    prompt += format_cot_example(curr, including_answer=False)
    return prompt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chains", type=int, default=5)
    parser.add_argument("--selected_subjects", "-sub", type=str, default="all")
    parser.add_argument("--save_dir", "-s", type=str, default="datasets")
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    full_test_df, full_val_df = load_mmlu_pro()
    all_subjects = sorted({each["category"] for each in full_test_df})
    if args.selected_subjects == "all":
        selected_subjects = all_subjects
    else:
        requested = [sub.strip().replace(" ", "_") for sub in args.selected_subjects.split(",") if sub.strip()]
        selected_subjects = []
        for subject in all_subjects:
            normalized_subject = subject.replace(" ", "_")
            if any(req in normalized_subject for req in requested):
                selected_subjects.append(subject)
        selected_subjects = sorted(set(selected_subjects))

    if not selected_subjects:
        raise ValueError("No subjects selected. Check '--selected_subjects'.")

    print("selected subjects:\n" + "\n".join(selected_subjects))

    for subject in selected_subjects:
        records = []
        test_df = select_by_category(full_test_df, subject)
        val_df = select_by_category(full_val_df, subject)
        for example in test_df:
            prompt = generate_cot_prompt(val_df, example, args.chains)
            records.append({
                "index": example["question_id"],
                "prompt": prompt,
                "answer": example["answer"],
            })
        output_path = save_dir / f"mmlu-pro-{subject}.jsonl"
        with output_path.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")
