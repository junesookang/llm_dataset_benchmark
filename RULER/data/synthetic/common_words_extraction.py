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

"""Generate jsonl data for the common words extraction benchmark.

Example:
python common_words_extraction.py --save_dir ./ --save_name vt \
    --tokenizer_path tokenizer.model --tokenizer_type nemo --max_seq_length 4096 \
    --tokens_to_generate 30 --num_samples 10 --random_seed 42 \
    --freq_cw 30 --freq_ucw 3 --num_cw 10 \
    --template "[INST] ... {context} ..."
"""

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import wonderwords
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.append(str(PACKAGE_ROOT))

from constants import TASKS
from manifest_utils import write_manifest
from tokenizer import select_tokenizer

logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser()
parser.add_argument("--save_dir", type=Path, required=True, help='dataset folder to save dataset')
parser.add_argument("--save_name", type=str, required=True, help='name of the save dataset jsonl file')
parser.add_argument("--subset", type=str, default='validation', help='Options: validation or test')
parser.add_argument("--tokenizer_path", type=str, required=True, help='path to the tokenizer model')
parser.add_argument("--tokenizer_type",  type=str, default='nemo', help='[Options] nemo, hf, openai.')
parser.add_argument("--max_seq_length", type=int, required=True, help='max sequence length including all input tokens and generated tokens.')
parser.add_argument("--tokens_to_generate", type=int, required=True, help='expected generated token amount.')
parser.add_argument("--num_samples", type=int, required=True, help='number of samples to generate')
parser.add_argument("--random_seed", type=int, default=42)
parser.add_argument("--template", type=str, default='', help='prompt template')
parser.add_argument("--remove_newline_tab", action='store_true', help='remove `\n` and `\t` in all strings.')

parser.add_argument("--freq_cw", type=int, default=30)
parser.add_argument("--freq_ucw", type=int, default=3)
parser.add_argument("--num_cw", type=int, default=10)
parser.add_argument("--num_fewshot", type=int, default=1)
parser.add_argument("--model_template_token", type=int, default=0, help='used for nemo skills, minus num of model template token')
args = parser.parse_args()


def load_word_list(file_name: str) -> List[str]:
    """Load a word list shipped with wonderwords (nounlist/adjectivelist/etc)."""
    return wonderwords.random_word._get_words_from_text_file(file_name)


def load_word_corpus(seed: int) -> Tuple[List[str], List[str]]:
    """Return the shuffled wonderwords corpus and fallback randle words."""
    nouns = load_word_list("nounlist.txt")
    adjs = load_word_list("adjectivelist.txt")
    verbs = load_word_list("verblist.txt")
    words = sorted(set(nouns + adjs + verbs))
    random.Random(seed).shuffle(words)
    logger.info("Loaded %d wonderwords", len(words))

    json_path = SCRIPT_DIR / "json" / "english_words.json"
    with json_path.open("r", encoding="utf-8") as file:
        randle_words = list(json.load(file).values())
    logger.info("Loaded %d randle words", len(randle_words))
    return words, randle_words


WONDER_WORDS, RANDLE_WORDS = load_word_corpus(args.random_seed)
RNG = random.Random(args.random_seed)
TOKENIZER = select_tokenizer(args.tokenizer_type, args.tokenizer_path)


def sample_word_set(num_words: int) -> Sequence[str]:
    """Sample a base set of words that will be replicated for the task."""
    corpus = WONDER_WORDS if num_words <= len(WONDER_WORDS) else RANDLE_WORDS
    if num_words > len(corpus):
        logger.warning("Requested %d words but corpus only has %d; sampling with replacement.", num_words, len(corpus))
        return RNG.choices(corpus, k=num_words)
    return RNG.sample(corpus, num_words)


def build_context(
    num_words: int,
    common_repeats: int,
    uncommon_repeats: int,
    common_count: int,
) -> Tuple[str, List[str]]:
    """Return the prompt context and solution words."""
    word_list_full = sample_word_set(num_words)
    common, uncommon = (
        list(word_list_full[:common_count]),
        list(word_list_full[common_count:]),
    )

    word_list = common * int(common_repeats) + uncommon * int(uncommon_repeats)
    RNG.shuffle(word_list)
    context = " ".join(f"{idx + 1}. {word}" for idx, word in enumerate(word_list))
    return context, common


def render_few_shot_examples(examples: Iterable[Tuple[str, List[str]]]) -> str:
    """Format few-shot examples using the provided template."""
    rendered = []
    for context, answer in examples:
        rendered.append(
            args.template.format(num_cw=args.num_cw, context=context, query="")
            + " "
            + " ".join(f"{idx + 1}. {word}" for idx, word in enumerate(answer))
        )
    return "\n".join(rendered)


def generate_input_output(num_words: int) -> Tuple[str, List[str]]:
    """Build the full prompt (few-shot + question) and solution words."""
    few_shots: List[Tuple[str, List[str]]] = []
    if args.max_seq_length < 4096:
        for _ in range(args.num_fewshot):
            few_shots.append(build_context(20, 3, 1, args.num_cw))
        context, answer = build_context(num_words, 6, 1, args.num_cw)
    else:
        for _ in range(args.num_fewshot):
            few_shots.append(build_context(40, 10, 3, args.num_cw))
        context, answer = build_context(num_words, args.freq_cw, args.freq_ucw, args.num_cw)

    few_shot_block = render_few_shot_examples(few_shots)
    question = args.template.format(num_cw=args.num_cw, context=context, query="")
    prompt = f"{few_shot_block}\n{question}" if few_shot_block else question
    return prompt, answer


def estimate_max_words(max_seq_length: int, incremental: int) -> int:
    """Estimate the upper bound of words that will fit within the token limit."""
    sample_input_text, _ = generate_input_output(4096)
    sample_tokens = len(TOKENIZER.text_to_tokens(sample_input_text)) or 1
    tokens_per_word = sample_tokens / 4096
    estimated = int(max_seq_length // max(tokens_per_word, 1e-6)) * 2
    return max(estimated, incremental * 2)


def find_optimal_word_count(
    max_seq_length: int,
    tokens_to_generate: int,
    incremental: int,
) -> int:
    """Binary search the number of words that fit into the sequence length."""
    lower_bound = incremental
    upper_bound = estimate_max_words(max_seq_length, incremental)
    optimal_num_words = incremental

    logger.info("Starting binary search between %d and %d", lower_bound, upper_bound)
    while lower_bound <= upper_bound:
        mid = (lower_bound + upper_bound) // 2
        input_text, _ = generate_input_output(mid)
        total_tokens = len(TOKENIZER.text_to_tokens(input_text)) + tokens_to_generate
        logger.info("Test words=%d -> tokens=%d/%d", mid, total_tokens, max_seq_length)

        if total_tokens <= max_seq_length:
            optimal_num_words = mid
            lower_bound = mid + 1
        else:
            upper_bound = mid - 1

    logger.info("Optimal haystack size: %d words", optimal_num_words)
    return optimal_num_words


def sys_word_pair_random(
    num_samples: int,
    max_seq_length: int,
    incremental: int = 10,
) -> List[dict]:
    """Generate dataset entries constrained by the tokenizer sequence length."""
    tokens_to_generate = args.tokens_to_generate
    effective_max_seq_length = max_seq_length - args.model_template_token

    num_words = find_optimal_word_count(effective_max_seq_length, tokens_to_generate, incremental)
    results: List[dict] = []

    for index in tqdm(range(num_samples)):
        used_words = num_words
        while True:
            try:
                input_text, answer = generate_input_output(used_words)
                length = len(TOKENIZER.text_to_tokens(input_text)) + tokens_to_generate
                if length > effective_max_seq_length:
                    raise ValueError(f"{length} exceeds max_seq_length {effective_max_seq_length}.")
                break
            except ValueError as exc:
                logger.debug("Input too long with %d words: %s", used_words, exc)
                if used_words > incremental:
                    used_words -= incremental
                    continue
                raise

        if args.remove_newline_tab:
            input_text = " ".join(input_text.replace("\n", " ").replace("\t", " ").strip().split())

        prefix_probe = TASKS["common_words_extraction"]["answer_prefix"][:10]
        answer_prefix_index = input_text.rfind(prefix_probe)
        if answer_prefix_index == -1:
            raise ValueError("Failed to locate answer prefix in the prompt.")
        answer_prefix = input_text[answer_prefix_index:]
        prompt_without_prefix = input_text[:answer_prefix_index]

        results.append(
            {
                "index": index,
                "input": prompt_without_prefix,
                "outputs": answer,
                "length": length,
                "length_w_model_temp": length + args.model_template_token,
                "answer_prefix": answer_prefix,
            }
        )

    return results


def main() -> None:
    save_file = args.save_dir / args.save_name / f"{args.subset}.jsonl"
    save_file.parent.mkdir(parents=True, exist_ok=True)

    dataset_entries = sys_word_pair_random(
        num_samples=args.num_samples,
        max_seq_length=args.max_seq_length,
    )

    write_manifest(save_file, dataset_entries)


if __name__ == "__main__":
    main()
