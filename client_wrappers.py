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
"""HTTP client wrappers for remote inference services."""

from __future__ import annotations

import abc
import json
import logging
import multiprocessing
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import requests
from tenacity import retry, stop_after_attempt, wait_random_exponential

LOGGER = logging.getLogger(__name__)


class Client(abc.ABC):
    """Base class for RESTful text generation clients."""

    def __init__(
        self,
        server_host: str,
        server_port: str = "5000",
        ssh_server: Optional[str] = None,
        ssh_key_path: Optional[str] = None,
        **generation_kwargs,
    ) -> None:
        self.server_host = server_host
        self.server_port = server_port
        self.ssh_server = os.getenv("SSH_SERVER", ssh_server)
        self.ssh_key_path = os.getenv("SSH_KEY_PATH", ssh_key_path)
        self.generation_kwargs = generation_kwargs
        self._session = requests.Session()

    @abc.abstractmethod
    def _single_call(self, **request_kwargs):
        raise NotImplementedError

    def __call__(self, prompt: str, **kwargs) -> Dict[str, Sequence[str]]:
        request = dict(self.generation_kwargs)
        request["prompts"] = [prompt]
        if kwargs.get("others"):
            request["others"] = kwargs["others"]

        outputs = self._single_call(**request)
        return {"text": outputs}

    @retry(wait=wait_random_exponential(min=15, max=60), stop=stop_after_attempt(3))
    def _send_request(self, payload: Dict, route: str = "generate"):
        url = f"http://{self.server_host}:{self.server_port}/{route}"
        headers = {"Content-Type": "application/json"}

        if self.ssh_server and self.ssh_key_path:
            import sshtunnel_requests

            tunnel_request = sshtunnel_requests.from_url(f"ssh://{self.ssh_server}:22", self.ssh_key_path)
            response = tunnel_request.put(url=url, data=json.dumps(payload), headers=headers)
        else:
            response = self._session.put(url=url, data=json.dumps(payload), headers=headers)

        response.raise_for_status()
        return response.json()

    def process_batch(self, prompts: List[str], **kwargs) -> List[Dict[str, Sequence[str]]]:
        max_workers = min(len(prompts), max(1, multiprocessing.cpu_count() * 4))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(self.__call__, prompt, **kwargs)
                for prompt in prompts
            ]
            return [future.result() for future in futures]


class VLLMClient(Client):
    def _single_call(
        self,
        prompts: Sequence[str],
        tokens_to_generate: int,
        temperature: float,
        top_p: float,
        top_k: int,
        random_seed: int,
        stop: Sequence[str],
        **_,
    ):
        request = {
            "prompt": prompts[0],
            "max_tokens": tokens_to_generate,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "stop": list(stop),
        }
        response = self._send_request(request)
        return response.get("text", [])


class SGLClient(Client):
    def _single_call(
        self,
        prompts: Sequence[str],
        tokens_to_generate: int,
        temperature: float,
        top_p: float,
        top_k: int,
        random_seed: int,
        stop: Sequence[str],
        **_,
    ):
        request = {
            "text": prompts[0],
            "sampling_params": {
                "max_new_tokens": tokens_to_generate,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "stop": list(stop),
            },
        }
        response = self._send_request(request)
        return response.get("text", [])


class OpenAIClient:
    """Wrapper supporting OpenAI."""

    MODEL_LENGTHS = {
        "gpt-4": 8192,
        "gpt-4-0613": 8192,
        "gpt-4-1106-preview": 128_000,
        "gpt-4-0125-preview": 128_000,
        "gpt-4-turbo-preview": 128_000,
        "gpt-3.5-turbo-0125": 16_385,
        "gpt-3.5-turbo-1106": 16_385,
        "gpt-3.5-turbo-0613": 4_096,
        "gpt-3.5-turbo": 16_385,
        "gpt-3.5-turbo-16k": 16_385,
        "gpt-3.5-turbo-16k-0613": 16_385,
        "gpt-4-32k": 32_768,
        "gpt-35-turbo-16k": 16_384,
    }

    def __init__(self, model_name: str, **generation_kwargs) -> None:
        self.openai_api_key = os.getenv("OPENAI_API_KEY")

        resolved_model = model_name

        self.model_name = resolved_model
        self.max_length = self.MODEL_LENGTHS.get(resolved_model, 16_385)
        self.generation_kwargs = generation_kwargs

        import tiktoken

        self.encoding = tiktoken.get_encoding("cl100k_base")
        self._create_client()

    def _create_client(self) -> None:
        from openai import OpenAI

        if self.openai_api_key:
            self.client = OpenAI(api_key=self.openai_api_key)
            return

    def _count_tokens(self, messages: Sequence[Dict[str, str]]) -> int:
        tokens_per_message = 3
        tokens_per_name = 1
        total = 0
        for message in messages:
            total += tokens_per_message
            for key, value in message.items():
                total += len(self.encoding.encode(value))
                if key == "name":
                    total += tokens_per_name
        return total + 3

    @retry(wait=wait_random_exponential(min=15, max=60), stop=stop_after_attempt(3))
    def _send_request(self, request: Dict):
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=request["msgs"],
                max_tokens=request["tokens_to_generate"],
                temperature=request["temperature"],
                seed=request["random_seed"],
                top_p=request["top_p"],
                stop=request["stop"],
            )
            return response
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.warning("OpenAI request failed: %s", exc)
            if getattr(exc, "status_code", None) == 401:
                self._create_client()
            raise

    def __call__(self, prompt: str) -> Dict[str, Sequence[str]]:
        messages = [{"role": "user", "content": prompt}]
        prompt_tokens = self._count_tokens(messages)
        request = dict(self.generation_kwargs)

        available = self.max_length - prompt_tokens
        if available < request["tokens_to_generate"]:
            LOGGER.info(
                "Reducing OpenAI max tokens from %d to %d due to context limit",
                request["tokens_to_generate"],
                available,
            )
            request["tokens_to_generate"] = max(1, available)

        request["msgs"] = messages
        response = self._send_request(request)
        text = response.choices[0].message.content if response.choices else ""
        return {"text": [text]}
