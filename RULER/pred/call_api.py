"""Batch inference utility for benchmark datasets."""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterator, List, Sequence

import yaml
from nemo.collections.asr.parts.utils.manifest_utils import read_manifest
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPT_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = SCRIPT_ROOT.parent

for candidate in (SCRIPT_ROOT, PROJECT_ROOT):
    if str(candidate) not in sys.path:
        sys.path.append(str(candidate))

logging.basicConfig(level=logging.INFO, force=True)
LOGGER = logging.getLogger(__name__)

SERVER_TYPES = (
    "trtllm",
    "vllm",
    "sglang",
    "openai",
    "gemini",
    "hf",
    "mamba",
)


class ServerAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        namespace.server_type = values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument("--data_dir", type=Path, required=True, help="Dataset jsonl directory.")
    parser.add_argument("--save_dir", type=Path, required=True, help="Destination directory for predictions.")
    parser.add_argument("--task", type=str, required=True, help="Task identifier within the benchmark.")
    parser.add_argument("--subset", type=str, default="validation", help="Subset to evaluate (validation|test).")
    parser.add_argument("--chunk_idx", type=int, default=0, help="Current chunk index when splitting datasets.")
    parser.add_argument("--chunk_amount", type=int, default=1, help="Total chunk count when splitting datasets.")

    # Server
    parser.add_argument("--server_type", default="nemo", action=ServerAction, choices=SERVER_TYPES)
    parser.add_argument("--server_host", type=str, default="127.0.0.1")
    parser.add_argument("--server_port", type=str, default="5000")
    parser.add_argument("--ssh_server", type=str)
    parser.add_argument("--ssh_key_path", type=str)
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="gpt-3.5-turbo",
        help="Model identifier for OpenAI/HF backends (API name or local checkpoint).",
    )

    # Inference
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=32)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--stop_words", type=str, default="")
    parser.add_argument("--sliding_window_size", type=int)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=1)

    args = parser.parse_args()
    args.stop_words = [word for word in args.stop_words.split(",") if word]
    if args.server_type in {"hf", "gemini"}:
        args.threads = 1
    return args


def load_task_config(args: argparse.Namespace) -> Dict:
    try:
        module = importlib.import_module(f"data.synthetic.constants")
    except ImportError as exc:
        raise ImportError(f"Module data.synthetic.constants not found.") from exc

    tasks_base = module.TASKS
    config_path = SCRIPT_DIR / f"../synthetic.yaml"
    with config_path.open("r", encoding="utf-8") as config_file:
        tasks_customized = yaml.safe_load(config_file)

    if args.task not in tasks_customized:
        raise ValueError(f"{args.task} is not found in config_tasks.yaml")

    config = dict(tasks_customized[args.task])
    config.update(tasks_base[config["task"]])
    return config


def resolve_prediction_path(args: argparse.Namespace) -> Path:
    if args.chunk_amount > 1:
        return args.save_dir / f"{args.task}-{args.chunk_idx}.jsonl"
    return args.save_dir / f"{args.task}.jsonl"


def load_dataset(task_file: Path, pred_file: Path) -> List[Dict]:
    if pred_file.exists():
        existing = {sample["index"] for sample in read_manifest(pred_file)}
        LOGGER.info("Resuming prediction; skipping %d existing entries.", len(existing))
        return [sample for sample in read_manifest(task_file) if sample["index"] not in existing]
    return read_manifest(task_file)


def build_llm(args: argparse.Namespace, tokens_to_generate: int):
    if args.server_type == "trtllm":
        from client_wrappers import TRTLLMClient

        return TRTLLMClient(
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
            max_attention_window_size=args.sliding_window_size,
        )

    if args.server_type == "vllm":
        from client_wrappers import VLLMClient

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

    if args.server_type == "sglang":
        from client_wrappers import SGLClient

        return SGLClient(
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

    if args.server_type == "openai":
        from client_wrappers import OpenAIClient

        return OpenAIClient(
            model_name=args.model_name_or_path,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            random_seed=args.random_seed,
            stop=args.stop_words,
            tokens_to_generate=tokens_to_generate,
        )

    if args.server_type == "gemini":
        from client_wrappers import GeminiClient

        return GeminiClient(
            model_name=args.model_name_or_path,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            random_seed=args.random_seed,
            stop=args.stop_words,
            tokens_to_generate=tokens_to_generate,
        )

    if args.server_type == "hf":
        from model_wrappers import create_huggingface_model

        return create_huggingface_model(
            name_or_path=args.model_name_or_path,
            stop=args.stop_words,
            do_sample=args.temperature > 0,
            repetition_penalty=1,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            max_new_tokens=tokens_to_generate,
        )

    if args.server_type == "mamba":
        from model_wrappers import MambaModel

        return MambaModel(
            name_or_path=args.model_name_or_path,
            repetition_penalty=1,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            stop=args.stop_words,
            max_new_tokens=tokens_to_generate,
        )

    raise RuntimeError(f"Unsupported server type {args.server_type}")


def batched(iterable: Sequence[Dict], batch_size: int) -> Iterator[List[Dict]]:
    batch: List[Dict] = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def extract_pred_text(pred) -> str:
    if isinstance(pred.get("text"), str):
        return pred["text"]
    if isinstance(pred.get("text"), list) and pred["text"]:
        return pred["text"][0]
    return ""


def generate_predictions(llm, batch: List[Dict]) -> List[Dict]:
    prompts = [sample["input"] for sample in batch]
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
                "input": sample["input"],
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

    config = load_task_config(args)
    task_file = args.data_dir / args.task / f"{args.subset}.jsonl"
    pred_file = resolve_prediction_path(args)
    pred_file.parent.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Predicting %s from %s into %s", args.task, task_file, pred_file)

    data = load_dataset(task_file, pred_file)
    if not data:
        LOGGER.info("No samples to process; exiting.")
        return

    llm = build_llm(args, config["tokens_to_generate"])

    # thread-safe writes
    write_lock = threading.Lock()
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

    elapsed = round((time.time() - start_time) / 60, 1)
    LOGGER.info("Completed predictions in %s minutes", elapsed)


if __name__ == "__main__":
    main()
