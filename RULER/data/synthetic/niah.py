"""Generate Needle-in-a-Haystack benchmark data."""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
import uuid
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import wonderwords
from nltk.tokenize import sent_tokenize
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

NEEDLE_TEMPLATE = "One of the special magic {type_needle_v} for {key} is: {value}."
ESSAY_PATH = SCRIPT_DIR / "json" / "PaulGrahamEssays.json"
DEPTHS = list(np.round(np.linspace(0, 100, num=40, endpoint=True)).astype(int))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--save_dir", type=Path, required=True, help="Directory where jsonl files are stored.")
    parser.add_argument("--save_name", type=str, required=True, help="Name of dataset folder.")
    parser.add_argument("--subset", type=str, default="validation", help="Dataset split: validation or test.")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Tokenizer model path.")
    parser.add_argument("--tokenizer_type", type=str, default="nemo", help="Tokenizer backend (nemo|hf|openai).")
    parser.add_argument("--max_seq_length", type=int, required=True, help="Max tokens (prompt + generation).")
    parser.add_argument("--tokens_to_generate", type=int, required=True, help="Expected generated token budget.")
    parser.add_argument("--num_samples", type=int, required=True, help="Number of dataset rows to create.")
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--template", type=str, default="", help="Prompt template.")
    parser.add_argument("--remove_newline_tab", action="store_true", help="Strip newlines/tabs from prompts.")

    parser.add_argument("--num_needle_k", type=int, default=1)
    parser.add_argument("--num_needle_v", type=int, default=1)
    parser.add_argument("--num_needle_q", type=int, default=1)
    parser.add_argument("--type_haystack", type=str, default="essay", help="Haystack type: noise|essay|needle.")
    parser.add_argument("--type_needle_k", type=str, default="words", help="Needle key type: numbers|words|uuids.")
    parser.add_argument("--type_needle_v", type=str, default="numbers", help="Needle value type: numbers|words|uuids.")
    parser.add_argument("--model_template_token", type=int, default=0, help="Offset for template token count.")

    args = parser.parse_args()
    args.num_needle_k = max(args.num_needle_k, args.num_needle_q)
    return args


def load_tokenizer(args: argparse.Namespace):
    return select_tokenizer(args.tokenizer_type, args.tokenizer_path)


def load_haystack_words(seed: int) -> List[str]:
    nouns = wonderwords.random_word._get_words_from_text_file("nounlist.txt")
    adjs = wonderwords.random_word._get_words_from_text_file("adjectivelist.txt")
    words = sorted({f"{adj}-{noun}" for adj in adjs for noun in nouns})
    random.Random(seed).shuffle(words)
    return words


def load_essay() -> List[str]:
    with ESSAY_PATH.open("r", encoding="utf-8") as essay_file:
        text = json.load(essay_file)["text"]
    return re.sub(r"\s+", " ", text).split(" ")


def random_number(num_digits: int = 7) -> str:
    lower_bound = 10 ** (num_digits - 1)
    upper_bound = 10**num_digits - 1
    return str(random.randint(lower_bound, upper_bound))


def random_word(words: Sequence[str]) -> str:
    return random.choice(words)


def random_uuid() -> str:
    return str(uuid.UUID(int=random.getrandbits(128), version=4))


def random_value(kind: str, words: Sequence[str]) -> str:
    if kind == "numbers":
        return random_number()
    if kind == "words":
        return random_word(words)
    if kind == "uuids":
        return random_uuid()
    raise ValueError(f"Unsupported needle type: {kind}")


def build_needles(
    args: argparse.Namespace,
    words: Sequence[str],
) -> Tuple[List[str], List[List[str]], List[str]]:
    keys: List[str] = []
    values: List[List[str]] = []
    needles: List[str] = []

    for _ in range(args.num_needle_k):
        key = random_value(args.type_needle_k, words)
        value_list = [
            random_value(args.type_needle_v, words)
            for _ in range(args.num_needle_v)
        ]
        keys.append(key)
        values.append(value_list)
        needles.extend(
            NEEDLE_TEMPLATE.format(
                type_needle_v=args.type_needle_v,
                key=key,
                value=value,
            )
            for value in value_list
        )

    random.shuffle(needles)
    return keys, values, needles


def inject_needles_into_essay(
    haystack_tokens: Sequence[str],
    needles: Sequence[str],
    length: int,
) -> str:
    if length <= len(haystack_tokens):
        text = " ".join(haystack_tokens[:length])
    else:
        repeats = (length + len(haystack_tokens) - 1) // len(haystack_tokens)
        text = " ".join((list(haystack_tokens) * repeats)[:length])

    sentences = sent_tokenize(text.strip())
    positions = [0] + sorted(
        int(len(sentences) * (depth / 100))
        for depth in random.sample(DEPTHS, len(needles))
    ) + [len(sentences)]

    segments: List[str] = []
    for idx in range(1, len(positions)):
        segments.append(" ".join(sentences[positions[idx - 1]:positions[idx]]))
        if idx - 1 < len(needles):
            segments.append(needles[idx - 1])
    return " ".join(segments)


def inject_needles_into_noise(
    base_sentence: str,
    needles: Sequence[str],
    length: int,
) -> str:
    sentences = [base_sentence] * length
    positions = sorted(random.sample(range(length), len(needles)), reverse=True)
    for pos, needle in zip(positions, needles):
        sentences.insert(pos, needle)
    return "\n".join(sentences)


def inject_needles_only(
    args: argparse.Namespace,
    words: Sequence[str],
    needles: Sequence[str],
    length: int,
) -> str:
    sentences = [
        NEEDLE_TEMPLATE.format(
            type_needle_v=args.type_needle_v,
            key=random_value(args.type_needle_k, words),
            value=random_value(args.type_needle_v, words),
        )
        for _ in range(length)
    ]
    positions = sorted(random.sample(range(length), len(needles)), reverse=True)
    for pos, needle in zip(positions, needles):
        sentences.insert(pos, needle)
    return "\n".join(sentences)


def build_context(
    args: argparse.Namespace,
    words: Sequence[str],
    haystack_tokens: Sequence[str],
    num_haystack: int,
) -> Tuple[str, List[str]]:
    keys, values, needles = build_needles(args, words)

    if args.type_haystack == "essay":
        context = inject_needles_into_essay(haystack_tokens, needles, num_haystack)
    elif args.type_haystack == "noise":
        context = inject_needles_into_noise(
            "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again.",
            needles,
            num_haystack,
        )
    elif args.type_haystack == "needle":
        context = inject_needles_only(args, words, needles, num_haystack)
    else:
        raise ValueError(f"Unsupported haystack type: {args.type_haystack}")

    indices = random.sample(range(args.num_needle_k), args.num_needle_q)
    queries = [keys[idx] for idx in indices]
    answers = [value for idx in indices for value in values[idx]]

    query_text = (
        ", ".join(queries[:-1]) + ", and " + queries[-1]
        if len(queries) > 1
        else queries[0]
    )

    template = args.template
    type_needle_v = args.type_needle_v
    if args.num_needle_q * args.num_needle_v == 1:
        template = (
            template.replace("Some", "A")
            .replace("are all", "is")
            .replace("are ", "is ")
            .replace("answers", "answer")
        )
        type_needle_v = type_needle_v.rstrip("s")

    input_text = template.format(
        type_needle_v=type_needle_v,
        context=context,
        query=query_text,
    )
    return input_text, answers


def estimate_haystack_length(
    args: argparse.Namespace,
    tokenizer,
    words: Sequence[str],
    haystack_tokens: Sequence[str],
    incremental: int,
) -> int:
    sample_input, _ = build_context(args, words, haystack_tokens, incremental)
    sample_tokens = len(tokenizer.text_to_tokens(sample_input))
    tokens_per_segment = sample_tokens / incremental

    max_seq = args.max_seq_length - args.model_template_token
    estimated = int((max_seq / max(tokens_per_segment, 1e-6)) * 3)
    LOGGER.info("Estimated %.1f tokens per haystack segment", tokens_per_segment)
    LOGGER.info("Binary search bounds: %d to %d", incremental, max(estimated, incremental * 2))

    lower = incremental
    upper = max(estimated, incremental * 2)
    optimal = incremental

    while lower <= upper:
        mid = (lower + upper) // 2
        input_text, _ = build_context(args, words, haystack_tokens, mid)
        total_tokens = len(tokenizer.text_to_tokens(input_text)) + args.tokens_to_generate
        LOGGER.info("Test haystack=%d -> tokens=%d/%d", mid, total_tokens, max_seq)

        if total_tokens <= max_seq:
            optimal = mid
            lower = mid + 1
        else:
            upper = mid - 1

    LOGGER.info("Selected haystack length: %d", optimal)
    return optimal


def generate_samples(
    args: argparse.Namespace,
    tokenizer,
    words: Sequence[str],
    haystack_tokens: Sequence[str],
) -> List[Dict]:
    incremental = 500 if args.type_haystack == "essay" else 25
    if args.type_haystack != "essay" and args.max_seq_length < 4096:
        incremental = 5

    num_haystack = estimate_haystack_length(args, tokenizer, words, haystack_tokens, incremental)
    tokens_budget = args.max_seq_length - args.model_template_token

    data: List[Dict] = []
    for idx in tqdm(range(args.num_samples), desc="Generating samples"):
        used_haystack = num_haystack

        while True:
            input_text, answers = build_context(args, words, haystack_tokens, used_haystack)
            length = len(tokenizer.text_to_tokens(input_text)) + args.tokens_to_generate
            if length <= tokens_budget:
                break
            used_haystack = max(used_haystack - incremental, incremental)

        if args.remove_newline_tab:
            input_text = " ".join(input_text.replace("\n", " ").replace("\t", " ").split())

        prefix_probe = TASKS["niah"]["answer_prefix"][:10]
        answer_prefix_index = input_text.rfind(prefix_probe)
        if answer_prefix_index == -1:
            raise ValueError("Failed to locate answer prefix.")

        answer_prefix = input_text[answer_prefix_index:]
        prompt = input_text[:answer_prefix_index]

        answer_pos = prompt.find(answers[0])
        token_position_answer = len(tokenizer.text_to_tokens(prompt[:answer_pos])) if answer_pos != -1 else -1

        data.append(
            {
                "index": idx,
                "input": prompt,
                "outputs": answers,
                "length": length,
                "length_w_model_temp": length + args.model_template_token,
                "answer_prefix": answer_prefix,
                "token_position_answer": token_position_answer,
            }
        )
    return data


def main() -> None:
    args = parse_args()
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    tokenizer = load_tokenizer(args)
    words = load_haystack_words(args.random_seed)
    haystack_tokens = load_essay() if args.type_haystack == "essay" else []

    dataset = generate_samples(args, tokenizer, words, haystack_tokens)

    save_file = args.save_dir / args.save_name / f"{args.subset}.jsonl"
    save_file.parent.mkdir(parents=True, exist_ok=True)
    write_manifest(save_file, dataset)


if __name__ == "__main__":
    main()
