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

Templates = {
    'base': "{system_prompt}{task_template}",

    'meta-llama': "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{task_template}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",

    'deepseek-ai': "<｜begin▁of▁sentence｜><｜User｜>{system_prompt}{task_template}<｜Assistant｜><think>\n",

    'Qwen': "<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{task_template}<|im_end|>\n<|im_start|>assistant\n\n",
}

BOS_TOKEN = {
    'base': "",

    'meta-llama': "<|begin_of_text|>",

    'deepseek-ai': "<｜begin▁of▁sentence｜>",

    'Qwen': "",
}

SYSTEM_PROMPT = {
    'base': "",

    'meta-llama': "You are a helpful assistant",

    'deepseek-ai': "",

    'Qwen': "You are a helpful assistant.",
}
