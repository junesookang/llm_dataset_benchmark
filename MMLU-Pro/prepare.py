import argparse
import json
import random
import re
from pathlib import Path

from datasets import load_dataset

choices = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
max_model_length = 4096
max_new_tokens = 2048
# random.seed(12345) # Commented out to ensure different shuffles on each run

INITIAL_PROMPT_PATH = Path(__file__).resolve().parent / "initial_prompt.txt"
INITIAL_PROMPT_TEXT = INITIAL_PROMPT_PATH.read_text(encoding="utf-8")


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
        each["options"] = options
        res_df.append(each)
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
        f"Question:\n{example['question']}\n",
        "Options:\n",
    ]
    prompt_parts.extend(
        f"{choices[i]}. {opt}\n" for i, opt in enumerate(example["options"])
    )

    if including_answer:
        cot_content = example["cot_content"].replace("A: Let's think step by step.",
                                                     "Answer: Let's think step by step.")
        #cot_content += "\n\n"
    else:
        cot_content = "Answer: Let's think step by step."

    #prompt_parts.append(cot_content)
    return "".join(prompt_parts), cot_content


def generate_cot_prompt(val_df, curr, k):
    assert k > 0
    subject = curr["category"]
    in_context_examples = select_by_category(val_df, subject, exclude_question=curr["question"])
    random.shuffle(in_context_examples)
    in_context_examples = in_context_examples[:k]
    prompts = []
    # initial prompt
    prompt = INITIAL_PROMPT_TEXT.replace("{$}", subject)
    user, assistant = format_cot_example(in_context_examples[0], including_answer=True)
    prompt += user
    prompts.append({"user": prompt, "assistant": assistant})
    # remaining in-context examples
    for example in in_context_examples[1:]:
        user, assistant = format_cot_example(example, including_answer=True)
        prompts.append({"user": user, "assistant": assistant})
    # last instruction without answer
    user, assistant = format_cot_example(curr, including_answer=False)
    prompts.append({"user": user, "assistant": assistant})
    return prompts


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
            prompts = generate_cot_prompt(val_df, example, args.chains)
            records.append({
                "index": example["question_id"],
                "prompt": prompts,
                "answer": example["answer"],
            })
        output_path = save_dir / f"mmlu-pro-{subject}.jsonl"
        with output_path.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")
