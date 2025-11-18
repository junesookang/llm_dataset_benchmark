# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the GSM8k dataset to parquet format
https://github.com/volcengine/verl/tree/main/examples/data_preprocess/gsm8k.py
"""

import argparse
import os
import re
from pathlib import Path

import datasets


def extract_solution(solution_str):
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
    assert solution is not None
    final_solution = solution.group(0)
    final_solution = final_solution.split("#### ")[1].replace(",", "")
    return final_solution


def load_hard_prompts():
    file = "prompt_hardest.txt"
    with open(file, "r") as f:
        hard_prompts = f.read().strip()
    hard_prompts = hard_prompts.split("\n\n")
    prompt = []
    for item in hard_prompts:
        splits = item.split("\nLet's think step by step\n")
        question = splits[0].replace("Question: ", "").strip()
        answer = splits[1].strip()
        prompt.append({"question": question, "answer": answer})
    return prompt


def format_cot_example(example, including_answer=True):

    if including_answer:
        cot_content, answer = example["answer"].split("\nThe answer is ")
        answer = answer.replace("\n", "").strip()
        cot_content = "Answer: Let's think step by step.\n" + cot_content.strip()
        cot_content += f"\nThe final answer is: $\\boxed{{{answer}}}$"
    else:
        cot_content = "Answer: Let's think step by step.\n"

    return f"Question: {example['question']}", cot_content


def generate_cot_prompt(in_context_examples, curr):
    prompts = []
    # initial prompt
    instruction_following = "Please reason step by step, and put your final answer within \\boxed{}."
    prompt = instruction_following + "\n\n"
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
    parser.add_argument("--save_dir", type=str, default="datasets")
    args = parser.parse_args()

    data_source = "openai/gsm8k"
    dataset = datasets.load_dataset("openai/gsm8k", "main")

    test_dataset = dataset["test"]
    hard_prompts = load_hard_prompts()

    instruction_following = "Please reason step by step, and put your final answer within \\boxed{}."

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = example.pop("question")

            answer_raw = example.pop("answer")

            curr = {
                "question": question_raw,
                "cot_content": answer_raw,
            }
            prompts = generate_cot_prompt(hard_prompts, curr)
            solution = extract_solution(answer_raw)
            data = {
                "index": idx,
                "prompt": prompts,
                "answer": solution,
            }
            return data

        return process_fn

    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    output_path = save_dir / "gsm8k.jsonl"
    test_dataset.to_json(str(output_path))
