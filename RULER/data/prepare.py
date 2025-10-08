# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

"""Prepare jsonl with fields `input` and `outputs`.
{
    "index": int,
    "input": str,
    "outputs": [str],
}

Example:
python prepare.py --save_dir ./ --task niah_single_1 \
    --tokenizer_path meta-llama/Llama-3.1-8B-Instruct --tokenizer_type hf \
    --max_seq_length 4096 --num_samples 10
"""
import argparse
import importlib
import os
import subprocess
import time
from pathlib import Path
import nltk
import yaml

from template import Templates

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

parser = argparse.ArgumentParser()
parser.add_argument("--save_dir", type=Path, required=True, help='dataset folder to save dataset')
parser.add_argument("--task", type=str, required=True, help='tasks in benchmark')
parser.add_argument("--subset", type=str, default='validation', help='Options: validation or test')
parser.add_argument("--tokenizer_path", type=str, required=True, help='path to the tokenizer model')
parser.add_argument("--tokenizer_type",  type=str, default='hf', help='[Options] hf, openai.')
parser.add_argument("--max_seq_length", type=int, required=True, help='max sequence length including all input tokens and generated tokens.')
parser.add_argument("--num_samples", type=int, default=500, help='maximum number of samples we want to test')
parser.add_argument("--random_seed", type=int, default=42)
parser.add_argument("--remove_newline_tab", action='store_true', help='remove `\n` and `\t` in all strings.')
parser.add_argument("--chunk_idx", type=int, default=0, help='index of current split chunk')
parser.add_argument("--chunk_amount", type=int, default=1, help='size of split chunk')
parser.add_argument("--prepare_for_ns", action='store_true')

args = parser.parse_args()


def main():
    start_time = time.time()
    curr_folder = Path(__file__).resolve().parent

    try:
        module = importlib.import_module(f"synthetic.constants")
    except ImportError as exc:
        raise ImportError(f"Module synthetic.constants not found.") from exc

    tasks_base = module.TASKS
    yaml_path = curr_folder / f"../synthetic.yaml"
    with yaml_path.open("r", encoding="utf-8") as yaml_file:
        tasks_customized = yaml.safe_load(yaml_file)

    if args.task not in tasks_customized:
        raise ValueError(f"{args.task} is not found in config_tasks.yaml")

    config = dict(tasks_customized.get(args.task, {}))
    config.update(tasks_base[config['task']])

    model_template_type = args.tokenizer_path.split('/')[0]
    if model_template_type not in Templates:
        available = ", ".join(Templates.keys())
        raise ValueError(f"{model_template_type} is not found in Templates. Available: {available}")

    model_template = Templates[model_template_type]

    if args.prepare_for_ns:
        from tokenizer import select_tokenizer

        TOKENIZER = select_tokenizer(args.tokenizer_type, args.tokenizer_path)
        model_template_token = len(TOKENIZER.text_to_tokens(model_template))
        model_template = Templates['base']
    else:
        model_template_token = None

    task_template = config['template']

    # Add answer prefix for all models
    answer_prefix = config['answer_prefix'] if 'answer_prefix' in config else ''

    config['template'] = model_template.format(task_template=task_template) + answer_prefix

    # Split task into multiple chunks
    chunks = [(args.num_samples // args.chunk_amount) + (1 if i < args.num_samples % args.chunk_amount else 0) for i in range(args.chunk_amount)]
    num_samples = chunks[args.chunk_idx]
    pre_samples = sum(chunks[:args.chunk_idx])

    random_seed = args.random_seed + args.chunk_idx

    save_file = args.save_dir / args.task / f"{args.subset}.jsonl"
    file_exists = False
    if save_file.exists():
        data = save_file.read_text(encoding="utf-8").splitlines()
        file_exists = len(data) == args.num_samples

    if file_exists:
        print(f"Skip preparing {args.task} with lines: {args.num_samples} to {save_file} (file exists)")
        return

    try:
        script = curr_folder / "synthetic" / f"{config['task']}.py"
        command = [
            "python",
            str(script),
            "--save_dir",
            str(args.save_dir),
            "--save_name",
            args.task,
            "--subset",
            args.subset,
            "--tokenizer_path",
            args.tokenizer_path,
            "--tokenizer_type",
            args.tokenizer_type,
            "--max_seq_length",
            str(args.max_seq_length),
            "--tokens_to_generate",
            str(config['tokens_to_generate']),
            "--num_samples",
            str(num_samples),
            "--random_seed",
            str(random_seed),
            "--template",
            config['template'],
        ]

        if config['task'] == 'qa':
            command.extend(["--pre_samples", str(pre_samples)])

        if args.remove_newline_tab:
            command.append("--remove_newline_tab")

        if args.prepare_for_ns and model_template_token is not None:
            command.extend(["--model_template_token", str(model_template_token)])

        for key, value in config['args'].items():
            command.extend([f"--{key}", str(value)])

        print("Executing:", " ".join(command))
        result = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        print("Output:\n", result.stdout)
        if result.stderr:
            print("Warnings:\n", result.stderr)
    except subprocess.CalledProcessError as exc:
        print("Error output:", exc.stderr)
    finally:
        elapsed_minutes = round((time.time() - start_time) / 60, 1)
        print(f"Prepare {args.task} with lines: {args.num_samples} to {save_file}")
        print(f"Used time: {elapsed_minutes} minutes")


if __name__ == '__main__':
    main()
