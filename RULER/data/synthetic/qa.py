"""Generate QA benchmark data from SQuAD or HotpotQA."""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
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

DOCUMENT_PROMPT = "Document {i}:\n{document}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--save_dir", type=Path, required=True, help="Directory to store generated jsonl files.")
    parser.add_argument("--save_name", type=str, required=True, help="Dataset subfolder name.")
    parser.add_argument("--subset", type=str, default="validation", help="Dataset split: validation or test.")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Tokenizer model path.")
    parser.add_argument("--tokenizer_type", type=str, default="nemo", help="Tokenizer backend (nemo|hf|openai).")
    parser.add_argument("--max_seq_length", type=int, required=True, help="Max tokens (prompt + generation).")
    parser.add_argument("--tokens_to_generate", type=int, required=True, help="Expected generated token budget.")
    parser.add_argument("--num_samples", type=int, required=True, help="Number of dataset rows to create.")
    parser.add_argument("--pre_samples", type=int, default=0, help="Number of already-generated samples to skip.")
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--template", type=str, required=True, help="Prompt template.")
    parser.add_argument("--remove_newline_tab", action="store_true", help="Strip newlines and tabs from prompts.")
    parser.add_argument("--model_template_token", type=int, default=0, help="Offset for template token count.")
    parser.add_argument("--dataset", type=str, required=True, help="Source dataset: squad or hotpotqa.")

    args = parser.parse_args()
    return args


def load_tokenizer(args: argparse.Namespace):
    return select_tokenizer(args.tokenizer_type, args.tokenizer_path)


def read_squad(path: Path) -> Tuple[List[Dict], List[str]]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    documents = sorted({paragraph["context"] for entry in data["data"] for paragraph in entry["paragraphs"]})
    doc_index = {doc: idx for idx, doc in enumerate(documents)}

    qas: List[Dict] = []
    for entry in data["data"]:
        context_indices = [doc_index[paragraph["context"]] for paragraph in entry["paragraphs"]]
        for paragraph in entry["paragraphs"]:
            base_idx = doc_index[paragraph["context"]]
            for qa in paragraph["qas"]:
                if qa.get("is_impossible"):
                    continue
                qas.append(
                    {
                        "query": qa["question"],
                        "outputs": [answer["text"] for answer in qa["answers"]],
                        "context": [base_idx],
                        "more_context": [idx for idx in context_indices if idx != base_idx],
                    }
                )
    return qas, documents


def read_hotpotqa(path: Path) -> Tuple[List[Dict], List[str]]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    documents = sorted({f"{title}\n{''.join(paragraphs)}" for entry in data for title, paragraphs in entry["context"]})
    doc_index = {doc: idx for idx, doc in enumerate(documents)}

    qas: List[Dict] = []
    for entry in data:
        qas.append(
            {
                "query": entry["question"],
                "outputs": [entry["answer"]],
                "context": [doc_index[f"{title}\n{''.join(paragraphs)}"] for title, paragraphs in entry["context"]],
            }
        )
    return qas, documents


def load_dataset(dataset: str) -> Tuple[List[Dict], List[str]]:
    if dataset.lower() == "squad":
        path = SCRIPT_DIR / "json" / "squad.json"
        return read_squad(path)
    if dataset.lower() == "hotpotqa":
        path = SCRIPT_DIR / "json" / "hotpotqa.json"
        return read_hotpotqa(path)
    raise ValueError(f"Unsupported dataset: {dataset}")


def assemble_context(
    qa_entry: Dict,
    num_docs: int,
    documents: Sequence[str],
) -> str:
    base_docs = list(qa_entry["context"])
    extra_docs = list(qa_entry.get("more_context", []))

    if num_docs < len(documents):
        needed = num_docs - len(base_docs)
        if needed <= 0:
            selected = base_docs[:num_docs]
        elif needed <= len(extra_docs):
            selected = base_docs + random.sample(extra_docs, needed)
        else:
            remaining = [idx for idx in range(len(documents)) if idx not in base_docs + extra_docs]
            additional = random.sample(remaining, max(0, needed - len(extra_docs)))
            selected = base_docs + extra_docs + additional
            selected = selected[:num_docs]
    else:
        repeats = (num_docs + len(documents) - 1) // len(documents)
        selected = list(range(len(documents))) * repeats
        selected = selected[:num_docs]

    selected_docs = [documents[idx] for idx in selected]
    random.shuffle(selected_docs)
    return "\n\n".join(DOCUMENT_PROMPT.format(i=i + 1, document=doc) for i, doc in enumerate(selected_docs))


def generate_input_output(
    args: argparse.Namespace,
    qas: Sequence[Dict],
    documents: Sequence[str],
    qa_index: int,
    num_docs: int,
) -> Tuple[str, List[str]]:
    entry = qas[qa_index]
    context = assemble_context(entry, num_docs, documents)
    input_text = args.template.format(context=context, query=entry["query"])
    return input_text, entry["outputs"]


def estimate_doc_count(
    args: argparse.Namespace,
    tokenizer,
    qas: Sequence[Dict],
    documents: Sequence[str],
    incremental: int,
) -> int:
    sample_input, _ = generate_input_output(args, qas, documents, 0, incremental)
    sample_tokens = len(tokenizer.text_to_tokens(sample_input))
    tokens_per_doc = sample_tokens / incremental

    tokens_budget = args.max_seq_length - args.model_template_token
    estimated = int((tokens_budget / max(tokens_per_doc, 1e-6)) * 3)
    lower = incremental
    upper = max(estimated, incremental * 2)

    LOGGER.info("Estimated %.1f tokens per doc; binary search bounds %d-%d", tokens_per_doc, lower, upper)

    optimal = incremental
    while lower <= upper:
        mid = (lower + upper) // 2
        input_text, _ = generate_input_output(args, qas, documents, 0, mid)
        total_tokens = len(tokenizer.text_to_tokens(input_text)) + args.tokens_to_generate

        LOGGER.info("Test docs=%d -> tokens=%d/%d", mid, total_tokens, tokens_budget)
        if total_tokens <= tokens_budget:
            optimal = mid
            lower = mid + 1
        else:
            upper = mid - 1

    LOGGER.info("Selected document count: %d", optimal)
    return optimal


def generate_samples(
    args: argparse.Namespace,
    tokenizer,
    qas: Sequence[Dict],
    documents: Sequence[str],
) -> List[Dict]:
    incremental = 10
    num_docs = estimate_doc_count(args, tokenizer, qas, documents, incremental)
    tokens_budget = args.max_seq_length - args.model_template_token

    samples: List[Dict] = []
    for idx in tqdm(range(args.num_samples), desc="Generating QA samples"):
        qa_idx = idx + args.pre_samples
        if qa_idx >= len(qas):
            raise IndexError("Requested more samples than available QAs.")

        used_docs = num_docs
        while True:
            input_text, answers = generate_input_output(args, qas, documents, qa_idx, used_docs)
            length = len(tokenizer.text_to_tokens(input_text)) + args.tokens_to_generate
            if length <= tokens_budget:
                break
            if used_docs <= incremental:
                raise ValueError("Unable to fit prompt within token budget.")
            used_docs -= incremental

        if args.remove_newline_tab:
            input_text = " ".join(input_text.replace("\n", " ").replace("\t", " ").split())

        prefix_probe = TASKS["qa"]["answer_prefix"][:10]
        answer_prefix_index = input_text.rfind(prefix_probe)
        if answer_prefix_index == -1:
            raise ValueError("Failed to locate answer prefix in prompt.")

        answer_prefix = input_text[answer_prefix_index:]
        prompt = input_text[:answer_prefix_index]

        samples.append(
            {
                "index": idx,
                "input": prompt,
                "outputs": answers,
                "length": length,
                "length_w_model_temp": length + args.model_template_token,
                "answer_prefix": answer_prefix,
            }
        )

    return samples


def main() -> None:
    args = parse_args()
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    tokenizer = load_tokenizer(args)
    qas, documents = load_dataset(args.dataset)

    dataset = generate_samples(args, tokenizer, qas, documents)

    save_file = args.save_dir / args.save_name / f"{args.subset}.jsonl"
    save_file.parent.mkdir(parents=True, exist_ok=True)
    write_manifest(save_file, dataset)


if __name__ == "__main__":
    main()
