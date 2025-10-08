# llm_dataset_benchmark

## Running the vLLM server

Launch the local inference server with the `serve_vllm.py` helper. Run the command
from the repository root (or adjust the path if needed) so the package imports
resolve correctly:

```bash
python -u serve_vllm.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --dtype bfloat16 \
  --tensor-parallel-size 4 \
  --disable-custom-all-reduce
```

Customize the flags for your deployment (for example, choose a different model
or tensor parallelism factor) while keeping `--model` pointing at a valid
checkpoint. The `-u` flag streams logs immediately to help diagnose loading
issues.

## Preparing benchmark datasets

Use `data/prepare.py` under `RULER` to materialize synthetic benchmark splits.
Run it from the repository root (or adjust the path) so imports resolve:

```bash
python data/prepare.py \
  --save_dir ./datasets \
  --task niah_single_1 \
  --tokenizer_path meta-llama/Llama-3.1-8B-Instruct \
  --tokenizer_type hf \
  --max_seq_length 4096 \
  --num_samples 10
```

Key flags:
- `--save_dir`: directory to write the processed jsonl files.
- `--task`: task key defined in `RULER/synthetic.yaml`.
- `--tokenizer_path`/`--tokenizer_type`: pick a tokenizer compatible with the model you will evaluate.
- `--max_seq_length`: keep within your hardware budget to avoid OOM during testing.
- `--num_samples`: limit rows when experimenting or split runs across chunks.

## Running predictions

After preparing a dataset, run inference with `pred/call_api.py`:

```bash
python pred/call_api.py \
  --data_dir "${DATA_DIR}" \
  --save_dir "${PRED_DIR}" \
  --task "${TASK}" \
  --server_type "${MODEL_FRAMEWORK}" \
  --model_name_or_path "${MODEL_PATH}" \
  --temperature "${TEMPERATURE}" \
  --top_k "${TOP_K}" \
  --top_p "${TOP_P}" \
  --batch_size "${BATCH_SIZE}"
```

Where:
- `DATA_DIR` points to the directory passed in `--save_dir` during preprocessing.
- `PRED_DIR` is where prediction jsonl files will be written.
- `TASK` must match an entry in `synthetic.yaml`.
- `MODEL_NAME_OR_PATH` identifies the model repo or checkpoint to query (HF repo id, local path, etc.).
- `SERVER_TYPE` selects the backend (`vllm`, `trtllm`, `hf`, `openai`, `gemini`, ...).
- Remaining parameters tune sampling and batching for your evaluation.

## Evaluating predictions

Generate summary metrics and submission CSVs with:

```bash
python eval/evaluate.py \
  --data_dir "${PRED_DIR}"
```

This script looks for `${TASK}.jsonl` files under `${PRED_DIR}`, aggregates chunked outputs, and writes `summary*.csv` plus `submission.csv` alongside the predictions.
