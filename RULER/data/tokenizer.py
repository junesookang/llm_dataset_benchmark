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
# limitations under the License.

import os
from typing import Callable, Dict, List, Protocol, Union
from tenacity import (
    retry,
    stop_after_attempt,
    wait_fixed,
    wait_random,
)


TokenList = Union[List[int], List[str]]


class Tokenizer(Protocol):
    """Lightweight tokenizer interface used by dataset preparation."""

    def text_to_tokens(self, text: str) -> TokenList:
        ...

    def tokens_to_text(self, tokens: TokenList) -> str:
        ...


def select_tokenizer(tokenizer_type: str, tokenizer_path: str) -> Tokenizer:
    """Return a tokenizer implementation based on the requested backend."""
    factories: Dict[str, Callable[[str], Tokenizer]] = {
        "hf": lambda path: HFTokenizer(model_path=path),
        "openai": lambda path: OpenAITokenizer(model_path=path),
        "gemini": lambda path: GeminiTokenizer(model_path=path),
    }

    try:
        return factories[tokenizer_type](tokenizer_path)
    except KeyError as exc:
        available = ", ".join(factories)
        raise ValueError(f"Unknown tokenizer_type '{tokenizer_type}'. Available: {available}") from exc


class HFTokenizer:
    """Tokenizer backed by Hugging Face AutoTokenizer."""

    def __init__(self, model_path: str) -> None:
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    def text_to_tokens(self, text: str) -> List[str]:
        return self.tokenizer.tokenize(text)

    def tokens_to_text(self, tokens: TokenList) -> str:
        return self.tokenizer.convert_tokens_to_string(tokens)


class OpenAITokenizer:
    """Tokenizer backed by tiktoken encodings."""

    def __init__(self, model_path: str = "cl100k_base") -> None:
        import tiktoken

        self.tokenizer = tiktoken.get_encoding(model_path)

    def text_to_tokens(self, text: str) -> List[int]:
        return self.tokenizer.encode(text)

    def tokens_to_text(self, tokens: TokenList) -> str:
        return self.tokenizer.decode(tokens)


class GeminiTokenizer:
    """Tokenizer backed by Gemini count-tokens API."""

    def __init__(self, model_path: str = "gemini-1.5-pro-latest") -> None:
        import google.generativeai as genai

        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        self.model = genai.GenerativeModel(model_path)

    @retry(wait=wait_fixed(60) + wait_random(0, 10), stop=stop_after_attempt(3))
    def text_to_tokens(self, text: str) -> List[int]:
        return list(range(self.model.count_tokens(text).total_tokens))

    def tokens_to_text(self, tokens: TokenList) -> str:
        raise NotImplementedError("Gemini API does not provide token-to-text conversion.")
