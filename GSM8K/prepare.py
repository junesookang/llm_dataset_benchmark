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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="datasets")
    args = parser.parse_args()

    data_source = "openai/gsm8k"
    dataset = datasets.load_dataset("openai/gsm8k", "main")

    test_dataset = dataset["test"]

    instruction_following = "Please reason step by step, and put your final answer within \\boxed{}."

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = example.pop("question")

            question = question_raw + " " + instruction_following

            answer_raw = example.pop("answer")
            solution = extract_solution(answer_raw)
            data = {
                "index": idx,
                "prompt": question,
                "answer": solution,
            }
            return data

        return process_fn

    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    output_path = save_dir / "gsm8k.jsonl"
    test_dataset.to_json(str(output_path))
