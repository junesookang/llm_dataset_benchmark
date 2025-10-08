"""Wrapper utilities for local model inference."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence

import torch

LOGGER = logging.getLogger(__name__)


class HuggingFaceModel:
    """Minimal text-generation interface backed by HF pipeline or raw model."""

    def __init__(
        self,
        *,
        tokenizer,
        stop_sequences: Sequence[str],
        generation_kwargs: Dict,
        pipeline_backend=None,
        model=None,
        device: Optional[str] = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.pipeline = pipeline_backend
        self.model = model
        self.device = device or "cpu"
        self.generation_kwargs = generation_kwargs
        self.stop_sequences = list(stop_sequences or [])

        if self.tokenizer.pad_token is None:
            # add pad token to allow batching (known issue for llama2-like models)
            self.tokenizer.padding_side = "left"
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def __call__(self, prompt: str, **kwargs) -> Dict[str, List[str]]:
        return self.process_batch([prompt], **kwargs)[0]

    def process_batch(self, prompts: List[str], **_: Dict) -> List[Dict[str, List[str]]]:
        if self.pipeline is not None:
            output = self.pipeline(text_inputs=prompts, **self.generation_kwargs)
            assert len(output) == len(prompts)
            generated_texts = [result[0]["generated_text"] for result in output]
        else:
            inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)
            generated_ids = self.model.generate(**inputs, **self.generation_kwargs)
            generated_texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        results = []
        for generated, prompt in zip(generated_texts, prompts):
            prompt_reference = prompt
            if self.pipeline is None:
                tokenized_prompt = self.tokenizer(prompt, return_tensors="pt", padding=True)
                prompt_reference = self.tokenizer.decode(
                    tokenized_prompt.input_ids[0], skip_special_tokens=True
                )

            completion = generated[len(prompt_reference) :] if generated.startswith(prompt_reference) else generated

            for stop in self.stop_sequences:
                completion = completion.split(stop)[0]

            results.append({"text": [completion]})

        return results


def create_huggingface_model(name_or_path: str, **generation_kwargs) -> HuggingFaceModel:
    """Factory that returns a HuggingFaceModel, preferring pipeline usage."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    tokenizer = AutoTokenizer.from_pretrained(name_or_path, trust_remote_code=True)
    stop_sequences = generation_kwargs.pop("stop", [])

    model_kwargs = None if "Yarn-Llama" in name_or_path else {"attn_implementation": "flash_attention_2"}

    pipeline_backend = None
    model = None
    device = "cpu"

    try:
        pipeline_backend = pipeline(
            "text-generation",
            model=name_or_path,
            tokenizer=tokenizer,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            model_kwargs=model_kwargs,
        )
        if hasattr(pipeline_backend.model, "device"):
            device = pipeline_backend.model.device
    except Exception as error:  # pylint: disable=broad-except
        LOGGER.debug("Falling back to AutoModelForCausalLM for %s (%s)", name_or_path, error)
        model = AutoModelForCausalLM.from_pretrained(
            name_or_path,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        device = getattr(model, "device", device)

    return HuggingFaceModel(
        tokenizer=tokenizer,
        stop_sequences=stop_sequences,
        generation_kwargs=generation_kwargs,
        pipeline_backend=pipeline_backend,
        model=model,
        device=device,
    )


class MambaModel:
    """Thin wrapper around the Mamba generation helper."""

    def __init__(self, name_or_path: str, **generation_kwargs) -> None:
        from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel # pyright: ignore[reportMissingImports]
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        self.device = "cuda"
        self.model = MambaLMHeadModel.from_pretrained(
            name_or_path,
            device=self.device,
            dtype=torch.bfloat16,
        )
        self.generation_kwargs = generation_kwargs
        self.stop_sequences = generation_kwargs.pop("stop", [])
        self.max_genlen = generation_kwargs.pop("max_new_tokens")

    def __call__(self, prompt: str, **kwargs) -> Dict[str, List[str]]:
        return self.process_batch([prompt], **kwargs)[0]

    def process_batch(self, prompts: List[str], **kwargs) -> List[Dict[str, List[str]]]:
        return [self._generate(prompt) for prompt in prompts]

    def _generate(self, prompt: str) -> Dict[str, List[str]]:
        tokens = self.tokenizer(prompt, return_tensors="pt")
        input_ids = tokens.input_ids.to(self.device)
        max_length = input_ids.shape[1] + self.max_genlen

        output = self.model.generate(
            input_ids=input_ids,
            max_length=max_length,
            cg=True,
            return_dict_in_generate=True,
            output_scores=True,
            enable_timing=False,
            **self.generation_kwargs,
        )
        completion = self.tokenizer.decode(output.sequences[0][input_ids.shape[1] :], skip_special_tokens=True)

        for stop in self.stop_sequences:
            completion = completion.split(stop)[0]

        return {"text": [completion]}
