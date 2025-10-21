"""Batch inference utility for benchmark datasets."""

from __future__ import annotations

import argparse
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Dict, Iterator, List, Optional, Sequence

from nemo.collections.asr.parts.utils.manifest_utils import read_manifest
from tqdm import tqdm

from transformers import GenerationConfig

from client_wrappers import Client, VLLMClient, OpenAIClient
from task_summary import TASKS, SUB_TASK_MAP


logging.basicConfig(level=logging.INFO, force=True)
LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run batched inference for a benchmark dataset.")

    # Data
    parser.add_argument("--save-dir", type=Path, required=True, help="Destination directory for predictions.")
    parser.add_argument("--task", type=str, required=True, help="Task identifier within the benchmark.")
    parser.add_argument(
        "--model-name",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
    )

    # Server
    parser.add_argument("--server-type", type=str, choices=["vllm", "openai"], default="vllm", help="Type of LLM server to use.")
    parser.add_argument("--server-host", type=str, default="127.0.0.1")
    parser.add_argument("--server-port", type=str, default="5000")
    parser.add_argument("--ssh-server", type=str, default=None)
    parser.add_argument("--ssh-key-path", type=str, default=None)

    # Inference
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--stop-words", type=str, default="")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)

    args = parser.parse_args()
    args.stop_words = [word for word in args.stop_words.split(",") if word]
    return args


def load_dataset(task_file: Path, pred_file: Path) -> List[Dict]:
    if pred_file.exists():
        existing_indices = {sample["index"] for sample in read_manifest(pred_file)}
        LOGGER.info("Resuming prediction; skipping %%d existing entries.", len(existing_indices))
        return [sample for sample in read_manifest(task_file) if sample["index"] not in existing_indices]
    return read_manifest(task_file)


def build_llm(args: argparse.Namespace, tokens_to_generate: int) -> Client:
    if args.server_type == "openai":
        return OpenAIClient(
            model_name=args.model_name,
            server_host=args.server_host,
            server_port=args.server_port,
            ssh_server=args.ssh_server,
            ssh_key_path=args.ssh_key_path,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            random_seed=args.random_seed,
            stop=args.stop_words,
            tokens_to_generate=tokens_to_generate,
        )
    else:
        return VLLMClient(
            server_host=args.server_host,
            server_port=args.server_port,
            ssh_server=args.ssh_server,
            ssh_key_path=args.ssh_key_path,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            random_seed=args.random_seed,
            stop=args.stop_words,
            tokens_to_generate=tokens_to_generate,
        )


def batched(items: Sequence[Dict], batch_size: int) -> Iterator[List[Dict]]:
    batch: List[Dict] = []
    for item in items:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def extract_pred_text(pred: Dict[str, Sequence[str]]) -> str:
    text = pred.get("text")
    if isinstance(text, str):
        return text
    if isinstance(text, list) and text:
        return text[0]
    return ""


def generate_predictions(llm: Client, batch: List[Dict]) -> List[Dict]:
    prompts = [sample["prompt"] for sample in batch]
    while True:
        try:
            predictions = llm.process_batch(prompts=prompts)
            break
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.exception("LLM call failed, retrying batch of size %d: %s", len(batch), exc)
            time.sleep(1)

    results = []
    for pred, sample in zip(predictions, batch):
        pred_text = extract_pred_text(pred)
        results.append(
            {
                "index": sample["index"],
                "pred": pred_text,
                "prompt": sample["prompt"],
                "outputs": sample.get("outputs", []),
                "others": sample.get("others", {}),
                "truncation": sample.get("truncation", -1),
                "length": sample.get("length", -1),
            }
        )
    return results


def main() -> None:
    args = parse_args()
    start_time = time.time()

    gen_config = GenerationConfig.from_pretrained(args.model_name)
    args.top_k = gen_config.top_k if gen_config.top_k is not None else 1
    args.top_p = gen_config.top_p if gen_config.top_p is not None else 1.0
    args.temperature = gen_config.temperature if gen_config.temperature is not None else 1.0
    task = SUB_TASK_MAP.get(args.task)
    dataset_dir = Path(TASKS.get(task)["dataset_dir"])
    filename = f"mmlu-pro-{args.task}.jsonl" if task == "mmlu-pro" else f"{args.task}.jsonl"
    task_file = dataset_dir / filename
    pred_file = args.save_dir / task / filename
    pred_file.parent.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Predicting %s from %s into %s", args.task, task_file, pred_file)

    data = load_dataset(task_file, pred_file)
    if not data:
        LOGGER.info("No samples to process; exiting.")
        return

    llm = build_llm(args, TASKS.get(task)["max_new_tokens"])

    write_lock = Lock()
    with pred_file.open("a", encoding="utf-8", buffering=1) as fout:
        with ThreadPoolExecutor(max_workers=args.threads) as executor:
            futures = {
                executor.submit(generate_predictions, llm, list(batch)): batch_idx
                for batch_idx, batch in enumerate(batched(data, args.batch_size))
            }

            pending: Dict[int, List[Dict]] = {}
            next_batch_to_write = 0

            for future in tqdm(as_completed(futures), total=len(futures), desc="Batches"):
                batch_idx = futures[future]
                pending[batch_idx] = future.result()

                while next_batch_to_write in pending:
                    records = sorted(pending.pop(next_batch_to_write), key=lambda item: item["index"])
                    with write_lock:
                        for record in records:
                            fout.write(json.dumps(record) + "\n")
                    next_batch_to_write += 1

    elapsed_minutes = round((time.time() - start_time) / 60, 1)
    LOGGER.info("Completed predictions in %s minutes", elapsed_minutes)


if __name__ == "__main__":
    main()

