"""Generate variable tracking benchmark data."""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import string
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
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

ESSAY_PATH = SCRIPT_DIR / "json" / "PaulGrahamEssays.json"
NOISE_SENTENCE = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--save_dir", type=Path, required=True, help="Directory to store generated jsonl files.")
    parser.add_argument("--save_name", type=str, required=True, help="Dataset subfolder name.")
    parser.add_argument("--subset", type=str, default="validation", help="Dataset split: validation or test.")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Tokenizer model path.")
    parser.add_argument("--tokenizer_type", type=str, default="nemo", help="Tokenizer backend (nemo|hf|openai).")
    parser.add_argument("--max_seq_length", type=int, required=True, help="Max tokens (prompt + generation).")
    parser.add_argument("--tokens_to_generate", type=int, default=120, help="Expected generated token budget.")
    parser.add_argument("--num_samples", type=int, required=True, help="Number of dataset rows to create.")
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--template", type=str, default="", help="Prompt template.")
    parser.add_argument("--remove_newline_tab", action="store_true", help="Strip newlines and tabs from prompts.")
    parser.add_argument("--model_template_token", type=int, default=0, help="Offset for template token count.")

    parser.add_argument("--type_haystack", type=str, default="noise", help="Haystack type: noise or essay.")
    parser.add_argument("--num_chains", type=int, default=1, help="Number of inserted variable chains.")
    parser.add_argument("--num_hops", type=int, default=4, help="Number of hops per chain.")
    parser.add_argument("--add_fewshot", action="store_true", default=False, help="Prepend a randomized ICL example.")

    return parser.parse_args()


def load_tokenizer(args: argparse.Namespace):
    return select_tokenizer(args.tokenizer_type, args.tokenizer_path)


def load_essay_tokens() -> List[str]:
    with ESSAY_PATH.open("r", encoding="utf-8") as file:
        text = json.load(file)["text"]
    return re.sub(r"\s+", " ", text).split(" ")


def generate_variables(count: int, length: int) -> List[str]:
    variables = set()
    while len(variables) < count:
        candidate = "".join(random.choices(string.ascii_uppercase, k=length))
        variables.add(candidate)
    return list(variables)


def build_chains(num_chains: int, num_hops: int, is_icl: bool) -> Tuple[List[List[str]], List[List[str]], str]:
    name_length = 3 if is_icl else 5
    variables = generate_variables(num_chains * (num_hops + 1), name_length)

    value = "12345" if is_icl else str(np.random.randint(10_000, 99_999))
    chains: List[List[str]] = []
    answers: List[List[str]] = []

    for idx in range(0, len(variables), num_hops + 1):
        chain_vars = variables[idx : idx + num_hops + 1]
        answers.append(chain_vars)

        statements = [f"VAR {chain_vars[0]} = {value}"]
        for hop in range(num_hops):
            statements.append(f"VAR {chain_vars[hop + 1]} = VAR {chain_vars[hop]}")
        chains.append(statements)

    return answers, chains, value


def interleave_chains(chains: Sequence[List[str]]) -> List[str]:
    queues = [list(chain) for chain in chains]
    result: List[str] = []
    while any(queues):
        candidates = [idx for idx, queue in enumerate(queues) if queue]
        choice = random.choice(candidates)
        result.append(queues[choice].pop(0))
    return result


def ensure_token_length(tokens: Sequence[str], target: int) -> List[str]:
    if len(tokens) >= target:
        return list(tokens[:target])
    repeats = (target + len(tokens) - 1) // len(tokens)
    extended = list(tokens) * repeats
    return extended[:target]


def compose_context(
    args: argparse.Namespace,
    haystack_tokens: Optional[Sequence[str]],
    num_segments: int,
    chains: Sequence[List[str]],
) -> str:
    if args.type_haystack == "essay":
        assert haystack_tokens is not None, "Essay haystack requires preloaded tokens."
        words = ensure_token_length(haystack_tokens, num_segments)
        sentences = sent_tokenize(" ".join(words).strip())
        chain_statements = interleave_chains(chains)

        if not sentences:
            sentences = [""]

        positions = sorted(
            int(len(sentences) * pos) for pos in np.random.rand(len(chain_statements))
        )

        segments: List[str] = []
        last_idx = 0
        for idx, statement in zip(positions, chain_statements):
            idx = min(idx, len(sentences))
            segments.append(" ".join(sentences[last_idx:idx]))
            segments.append(statement.strip() + ".")
            last_idx = idx
        segments.append(" ".join(sentences[last_idx:]))
        context = " ".join(segment for segment in segments if segment)

    elif args.type_haystack == "noise":
        sentences = [NOISE_SENTENCE] * num_segments
        for chain in chains:
            positions = sorted(random.sample(range(len(sentences) + 1), len(chain)))
            offset = 0
            for pos, statement in zip(positions, chain):
                sentences.insert(pos + offset, statement)
                offset += 1
        context = "\n".join(sentences)
    else:
        raise ValueError(f"Unsupported haystack type: {args.type_haystack}")

    return context.replace(". \n", ".\n")


def build_prompt(
    args: argparse.Namespace,
    context: str,
    query_value: str,
    num_hops: int,
) -> str:
    template = args.template or (
        TASKS["variable_tracking"]["template"] + TASKS["variable_tracking"]["answer_prefix"]
    )
    return template.format(
        context=context,
        query=query_value,
        num_v=num_hops + 1,
    )


def generate_example(
    args: argparse.Namespace,
    haystack_tokens: Optional[Sequence[str]],
    num_segments: int,
    num_chains: int,
    num_hops: int,
    is_icl: bool,
) -> Tuple[str, List[str]]:
    answers, chains, value = build_chains(num_chains, num_hops, is_icl=is_icl)
    context = compose_context(args, haystack_tokens, num_segments, chains)
    prompt = build_prompt(args, context, value, num_hops)
    return prompt, answers[0]


def randomize_icl_text(text: str, num_hops: int) -> str:
    tokens = text.strip().split()
    target_vars = tokens[-(num_hops + 1) :]
    replacements = {
        var: "".join(random.choices(string.ascii_uppercase, k=len(var))) for var in target_vars
    }

    randomized = text
    for original, replacement in replacements.items():
        randomized = randomized.replace(original, replacement)

    if "12345" in randomized:
        new_value = str(np.random.randint(10_000, 99_999))
        randomized = randomized.replace("12345", new_value, 1)

    return randomized


def determine_incremental(args: argparse.Namespace, has_icl: bool) -> int:
    if has_icl:
        return 500 if args.type_haystack == "essay" else 10
    return 50 if args.type_haystack == "essay" else 5


def estimate_segment_count(
    args: argparse.Namespace,
    tokenizer,
    haystack_tokens: Optional[Sequence[str]],
    num_chains: int,
    num_hops: int,
    incremental: int,
    example_tokens: int,
    tokens_to_generate: int,
    use_icl_template: bool,
) -> int:
    sample_prompt, _ = generate_example(
        args,
        haystack_tokens,
        incremental,
        num_chains,
        num_hops,
        is_icl=use_icl_template,
    )
    tokens_per_segment = len(tokenizer.text_to_tokens(sample_prompt)) / max(incremental, 1)

    tokens_budget = args.max_seq_length - args.model_template_token
    estimated = int((tokens_budget / max(tokens_per_segment, 1e-6)) * 3)
    lower = incremental
    upper = max(estimated, incremental * 2)

    LOGGER.info("Estimated %.1f tokens per segment; search bounds %d-%d", tokens_per_segment, lower, upper)

    optimal = incremental
    while lower <= upper:
        mid = (lower + upper) // 2
        prompt, _ = generate_example(
            args,
            haystack_tokens,
            mid,
            num_chains,
            num_hops,
            is_icl=use_icl_template,
        )
        total_tokens = len(tokenizer.text_to_tokens(prompt)) + example_tokens + tokens_to_generate
        LOGGER.info("Test segments=%d -> tokens=%d/%d", mid, total_tokens, tokens_budget)

        if total_tokens <= tokens_budget:
            optimal = mid
            lower = mid + 1
        else:
            upper = mid - 1

    LOGGER.info("Selected segment count: %d", optimal)
    return optimal


def render_icl_example(example: Dict) -> str:
    base = example["input"].strip()
    outputs = " ".join(example["outputs"])
    return f"{base} {outputs}\n"


def generate_samples(
    args: argparse.Namespace,
    tokenizer,
    haystack_tokens: Optional[Sequence[str]],
    num_samples: int,
    num_chains: int,
    num_hops: int,
    icl_example: Optional[Dict],
    add_fewshot: bool,
    include_generation: bool,
    final_output: bool,
) -> List[Dict]:
    tokens_to_generate = args.tokens_to_generate if include_generation else 0

    icl_base_text = None
    example_tokens = 0
    if add_fewshot and icl_example is not None:
        icl_base_text = render_icl_example(icl_example)
        example_tokens = len(tokenizer.text_to_tokens(icl_base_text))

    incremental = determine_incremental(args, has_icl=icl_example is not None)
    use_icl_template = add_fewshot and icl_example is None

    num_segments = estimate_segment_count(
        args,
        tokenizer,
        haystack_tokens,
        num_chains,
        num_hops,
        incremental,
        example_tokens,
        tokens_to_generate,
        use_icl_template,
    )

    tokens_budget = args.max_seq_length - args.model_template_token
    samples: List[Dict] = []

    for idx in tqdm(range(num_samples), desc="Generating variable tracking samples"):
        used_segments = num_segments

        while True:
            prompt, answers = generate_example(
                args,
                haystack_tokens,
                used_segments,
                num_chains,
                num_hops,
                is_icl=use_icl_template,
            )

            prompt_with_icl = prompt
            if add_fewshot and icl_base_text is not None:
                randomized_icl = randomize_icl_text(icl_base_text, num_hops).strip()
                prompt_with_icl = f"{randomized_icl}\n\n{prompt}"

            candidate_prompt = prompt_with_icl
            if args.remove_newline_tab:
                candidate_prompt = " ".join(
                    candidate_prompt.replace("\n", " ").replace("\t", " ").split()
                )

            total_tokens = len(tokenizer.text_to_tokens(candidate_prompt)) + tokens_to_generate

            if total_tokens <= tokens_budget:
                break

            used_segments = max(used_segments - incremental, incremental)

        answer_prefix = ""
        trimmed_prompt = candidate_prompt
        if final_output:
            prefix_probe = TASKS["variable_tracking"]["answer_prefix"][:10]
            answer_prefix_index = candidate_prompt.rfind(prefix_probe)
            if answer_prefix_index == -1:
                raise ValueError("Failed to locate answer prefix in prompt.")
            answer_prefix = candidate_prompt[answer_prefix_index:]
            trimmed_prompt = candidate_prompt[:answer_prefix_index]

        record: Dict[str, object] = {
            "index": idx,
            "input": trimmed_prompt,
            "outputs": answers,
            "length": total_tokens,
        }
        if final_output:
            record.update(
                {
                    "length_w_model_temp": total_tokens + args.model_template_token,
                    "answer_prefix": answer_prefix,
                }
            )
        samples.append(record)

    return samples


def main() -> None:
    args = parse_args()
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    haystack_tokens = load_essay_tokens() if args.type_haystack == "essay" else None
    tokenizer = load_tokenizer(args)

    icl_example = None
    if args.add_fewshot:
        icl_example = generate_samples(
            args,
            tokenizer,
            haystack_tokens,
            num_samples=1,
            num_chains=args.num_chains,
            num_hops=args.num_hops,
            icl_example=None,
            add_fewshot=False,
            include_generation=False,
            final_output=False,
        )[0]
        LOGGER.info("ICL example generated: %s", icl_example)

    dataset = generate_samples(
        args,
        tokenizer,
        haystack_tokens,
        num_samples=args.num_samples,
        num_chains=args.num_chains,
        num_hops=args.num_hops,
        icl_example=icl_example,
        add_fewshot=args.add_fewshot,
        include_generation=True,
        final_output=True,
    )

    save_file = args.save_dir / args.save_name / f"{args.subset}.jsonl"
    save_file.parent.mkdir(parents=True, exist_ok=True)
    write_manifest(save_file, dataset)


if __name__ == "__main__":
    main()
