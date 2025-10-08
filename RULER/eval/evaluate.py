"""Evaluate RULER predictions and derive submission artifacts."""

from __future__ import annotations

import argparse
import importlib
import logging
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd
import yaml
from nemo.collections.asr.parts.utils.manifest_utils import read_manifest, write_manifest
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, force=True)
LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, required=True, help="Directory holding prediction jsonl files.")
    parser.add_argument("--verbose", type=int, default=0, help="Show N example predictions for debugging.")
    return parser.parse_args()


def load_task_configs(config_dir: Path) -> Dict[str, Dict]:
    module = importlib.import_module(f"synthetic.constants")
    tasks_base = module.TASKS

    yaml_path = config_dir / f"synthetic.yaml"
    with yaml_path.open("r", encoding="utf-8") as yaml_file:
        tasks_customized = yaml.safe_load(yaml_file)

    for task_name, config in tasks_customized.items():
        config.update(tasks_base[config["task"]])
    LOGGER.info("Loaded tasks: %s", list(tasks_customized.keys()))
    return tasks_customized


def postprocess_prediction(text: str) -> str:
    np_pattern = re.compile(r"[\x00-\x1f]")
    return np_pattern.sub("\n", text.strip()).strip()


def extract_predictions(
    prediction_path: Path,
    config: Dict,
    input_field: str = "input",
    references_field: str = "outputs",
    prediction_field: str = "pred",
    metadata_field: str = "others",
) -> Tuple[List[str], List[str], List[List[str]], List[str]]:
    records = read_manifest(str(prediction_path))

    inputs: List[str] = []
    predictions: List[str] = []
    references: List[List[str]] = []
    indices: List[str] = []

    for record in tqdm(records, desc=f"Reading {prediction_path.name}"):
        inputs.append(record[input_field])
        predictions.append(postprocess_prediction(record[prediction_field]))
        references.append(record.get(references_field, [record.get("output", "")]))
        indices.append(record.get(metadata_field, {}).get("id", record["index"]))

    return inputs, predictions, references, indices


def run_task_evaluation(
    prediction_path: Path,
    config: Dict,
    verbose: int,
) -> Tuple[float, str, List[str], List[str]]:
    inputs, predictions, references, indices = extract_predictions(prediction_path, config)

    null_count = sum(len(pred) == 0 for pred in predictions)
    null_summary = f"{null_count}/{len(predictions)}"

    if references and references[0][0] is not None:
        score = config["metric_fn"](predictions, references)
    else:
        score = 0.0

    if verbose:
        LOGGER.info("Sample predictions for %s", prediction_path.stem)
        for idx, (inp, ref, pred) in enumerate(zip(inputs, references, predictions)):
            LOGGER.info("Input: %s\nReference: %s\nPrediction: %s\n%s", inp, ref, pred, "-" * 40)
            if idx >= verbose:
                break

    return score, null_summary, predictions, indices


def aggregate_chunked_predictions(folder: Path) -> None:
    jsonl_files = [path for path in folder.iterdir() if path.suffix == ".jsonl"]
    chunk_groups: Dict[str, List[Path]] = defaultdict(list)

    for path in jsonl_files:
        match = re.match(r"(.*)-\d+\.jsonl$", path.name)
        if match:
            chunk_groups[match.group(1)].append(path)

    for task, files in chunk_groups.items():
        combined: List[Dict] = []
        for path in sorted(files):
            combined.extend(read_manifest(str(path)))
            path.unlink()
        output_path = folder / f"{task}.jsonl"
        write_manifest(str(output_path), combined)
        LOGGER.info("Aggregated %s chunked files into %s", len(files), output_path.name)


def write_summary(results: Dict[str, Dict[str, str]], data_dir: Path) -> None:
    tasks = list(results.keys())
    rows = [
        ["Tasks"] + tasks,
        ["Score"] + [results[task]["score"] for task in tasks],
        ["Nulls"] + [results[task]["nulls"] for task in tasks],
    ]
    output_path = data_dir / ("summary.csv" if len(tasks) > 1 else f"summary-{tasks[0]}.csv")
    pd.DataFrame(rows).to_csv(output_path, index=False)
    LOGGER.info("Saved summary to %s", output_path)


def write_submission(results: Dict[str, Dict[str, List[str]]], data_dir: Path) -> None:
    rows: List[Dict[str, str]] = []
    for task, info in results.items():
        for idx, prediction in zip(info["indices"], info["predicts"]):
            rows.append({"Task": task, "ID": idx, "Prediction": prediction})
    output_path = data_dir / "submission.csv"
    pd.DataFrame(rows, columns=["Task", "ID", "Prediction"]).to_csv(output_path, index=False)
    LOGGER.info("Saved submission payload to %s", output_path)


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    config_dir = Path(__file__).resolve().parent

    tasks = load_task_configs(config_dir.parent)
    aggregate_chunked_predictions(data_dir)

    available_files = {path.name for path in data_dir.glob("*.jsonl")}
    eval_results: Dict[str, Dict[str, str]] = {}
    submission_payload: Dict[str, Dict[str, List[str]]] = {}

    for task_name, config in tasks.items():
        prediction_file = data_dir / f"{task_name}.jsonl"
        if prediction_file.name not in available_files:
            LOGGER.warning("Prediction file %s not found; skipping.", prediction_file.name)
            continue

        LOGGER.info("Evaluating task %s", task_name)
        score, nulls, predictions, indices = run_task_evaluation(
            prediction_path=prediction_file,
            config=config,
            verbose=args.verbose,
        )
        eval_results[task_name] = {"score": score, "nulls": nulls}
        submission_payload[task_name] = {"predicts": predictions, "indices": indices}

    if not eval_results:
        LOGGER.warning("No evaluation results produced; exiting without artifacts.")
        return

    write_summary(eval_results, data_dir)
    write_submission(submission_payload, data_dir)


if __name__ == "__main__":
    main()
