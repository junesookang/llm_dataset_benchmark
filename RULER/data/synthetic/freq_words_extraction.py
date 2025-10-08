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
# limitations under the License
"""Create jsonl data for the frequent words extraction benchmark.

Example:
python freq_words_extraction.py --save_dir ./ --save_name vt \
    --tokenizer_path tokenizer.model --tokenizer_type nemo --max_seq_length 4096 \
    --tokens_to_generate 30 --num_samples 10 --random_seed 42 --alpha 2.0 \
    --template "[INST] ... {context} ..."
"""

import argparse
import logging
import random
import string
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
from scipy.special import zeta
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.append(str(PACKAGE_ROOT))

from constants import TASKS  # noqa: E402
from manifest_utils import write_manifest  # noqa: E402
from tokenizer import select_tokenizer  # noqa: E402

logging.basicConfig(level=logging.INFO, force=True)
LOGGER = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--save_dir", type=Path, required=True, help="Dataset output directory.")
parser.add_argument("--save_name", type=str, required=True, help="Name of the dataset folder.")
parser.add_argument("--subset", type=str, default="validation", help="Subset name (validation|test).")
parser.add_argument("--tokenizer_path", type=str, required=True, help="Tokenizer model path.")
parser.add_argument("--tokenizer_type", type=str, default="nemo", help="Tokenizer backend (nemo|hf|openai).")
parser.add_argument(
    "--max_seq_length",
    type=int,
    required=True,
    help="Max tokens (prompt + generation) allowed by the tokenizer.",
)
parser.add_argument("--tokens_to_generate", type=int, default=50, help="Expected generated token budget.")
parser.add_argument("--num_samples", type=int, required=True, help="Number of dataset rows to create.")
parser.add_argument("--random_seed", type=int, default=42, help="Seed for deterministic sampling.")
parser.add_argument("--template", type=str, default="", help="Prompt template with `{context}` placeholder.")
parser.add_argument("--remove_newline_tab", action="store_true", help="Strip newlines and tabs from prompts.")
parser.add_argument("--coded_wordlen", type=int, default=6, help="Length of synthetic coded words.")
parser.add_argument("--vocab_size", type=int, default=-1, help="Synthetic vocabulary size (auto when -1).")
parser.add_argument("--alpha", type=float, default=2.0, help="Zipf alpha parameter controlling frequency skew.")
parser.add_argument("--model_template_token", type=int, default=0, help="Template token count to subtract.")

args = parser.parse_args()
random.seed(args.random_seed)
np.random.seed(args.random_seed)

TOKENIZER = select_tokenizer(args.tokenizer_type, args.tokenizer_path)


def build_vocab(vocab_size: int, word_length: int) -> List[str]:
    """Create a synthetic vocabulary of distinct coded words."""
    vocab: set[str] = set()
    while len(vocab) < vocab_size:
        vocab.add("".join(random.choices(string.ascii_lowercase, k=word_length)))

    vocab_list = sorted(vocab)
    random.Random(args.random_seed).shuffle(vocab_list)
    if vocab_list:
        vocab_list[0] = "..."  # promote noise token
    return vocab_list


def render_prompt(words: List[str]) -> str:
    """Render the prompt template with the provided word list."""
    return args.template.format(context=" ".join(words), query="")


def sample_word_list(vocab: List[str], num_words: int, alpha: float) -> Tuple[List[str], List[str]]:
    """Sample a list of words following a Zipf-like distribution."""
    ranks = np.arange(1, len(vocab) + 1)
    counts = num_words * (ranks ** -alpha) / zeta(alpha)
    sampled_words = [word for word, count in zip(vocab, counts.astype(int)) for _ in range(count)]
    random.Random(args.random_seed).shuffle(sampled_words)
    return sampled_words, vocab[1:4]


def generate_input_output(
    max_tokens: int,
    *,
    num_words: int = -1,
    coded_wordlen: int,
    vocab_size: int,
    incremental: int,
    alpha: float,
) -> Tuple[str, List[str], int]:
    """Construct a prompt constrained by the tokenizer length."""
    vocab = build_vocab(vocab_size, coded_wordlen)
    step = max(incremental, 1)

    def build_prompt(word_count: int) -> Tuple[str, List[str]]:
        sampled_words, answers = sample_word_list(vocab, word_count, alpha)
        prompt = render_prompt(sampled_words)
        return prompt, answers

    if num_words > 0:
        current_words = num_words
        prompt, answers = build_prompt(current_words)
        while len(TOKENIZER.text_to_tokens(prompt)) > max_tokens and current_words > step:
            current_words -= step
            prompt, answers = build_prompt(current_words)
        while len(TOKENIZER.text_to_tokens(prompt)) > max_tokens and current_words > 1:
            current_words -= 1
            prompt, answers = build_prompt(current_words)
    else:
        current_words = max(max_tokens // coded_wordlen, step)
        prompt, answers = build_prompt(current_words)
        while len(TOKENIZER.text_to_tokens(prompt)) < max_tokens:
            current_words += step
            prompt, answers = build_prompt(current_words)
        while len(TOKENIZER.text_to_tokens(prompt)) > max_tokens and current_words > step:
            current_words -= step
            prompt, answers = build_prompt(current_words)
        while len(TOKENIZER.text_to_tokens(prompt)) > max_tokens and current_words > 1:
            current_words -= 1
            prompt, answers = build_prompt(current_words)

    if len(TOKENIZER.text_to_tokens(prompt)) > max_tokens:
        raise ValueError("Failed to fit prompt within the max token budget.")

    return prompt, answers, current_words


def sys_kwext(num_samples: int, max_seq_length: int, incremental: int = 10) -> List[dict]:
    """Generate dataset entries for frequent word extraction."""
    tokens_to_generate = args.tokens_to_generate
    effective_max_seq_length = max_seq_length - args.model_template_token - tokens_to_generate
    if effective_max_seq_length <= 0:
        raise ValueError("Effective max sequence length must be positive.")

    vocab_size = max_seq_length // 50 if args.vocab_size == -1 else args.vocab_size
    estimation_increment = max(effective_max_seq_length // 32, 1)

    _, _, num_example_words = generate_input_output(
        effective_max_seq_length,
        coded_wordlen=args.coded_wordlen,
        vocab_size=vocab_size,
        incremental=estimation_increment,
        alpha=args.alpha,
    )
    LOGGER.info("Using %d words per example", num_example_words)

    results: List[dict] = []
    for index in tqdm(range(num_samples), desc="Preparing samples"):
        prompt, answers, _ = generate_input_output(
            effective_max_seq_length,
            num_words=num_example_words,
            coded_wordlen=args.coded_wordlen,
            vocab_size=vocab_size,
            incremental=estimation_increment,
            alpha=args.alpha,
        )

        length = len(TOKENIZER.text_to_tokens(prompt)) + tokens_to_generate

        if args.remove_newline_tab:
            prompt = " ".join(prompt.replace("\n", " ").replace("\t", " ").strip().split())

        prefix_probe = TASKS["freq_words_extraction"]["answer_prefix"][:10]
        answer_prefix_index = prompt.rfind(prefix_probe)
        if answer_prefix_index == -1:
            raise ValueError("Failed to locate answer prefix in the prompt.")
        answer_prefix = prompt[answer_prefix_index:]
        prompt_without_prefix = prompt[:answer_prefix_index]

        results.append(
            {
                "index": index,
                "input": prompt_without_prefix,
                "outputs": answers,
                "length": length,
                "length_w_model_temp": length + args.model_template_token,
                "answer_prefix": answer_prefix,
            }
        )

    return results


def main() -> None:
    save_file = args.save_dir / args.save_name / f"{args.subset}.jsonl"
    save_file.parent.mkdir(parents=True, exist_ok=True)
    dataset_entries = sys_kwext(
        num_samples=args.num_samples,
        max_seq_length=args.max_seq_length,
        incremental=10,
    )

    write_manifest(save_file, dataset_entries)


if __name__ == "__main__":
    main()
