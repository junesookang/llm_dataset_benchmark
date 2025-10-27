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
    'base': "{task_template}",

    'meta-llama': "<|start_header_id|>user<|end_header_id|>\n\n{task_template}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",

    'deepseek-ai': "<｜User｜>{task_template}<｜Assistant｜>",

    'Qwen': "<|im_start|>user\n{task_template}<|im_end|>\n<|im_start|>assistant\n",
}

THINK_TOKEN = "<think>\n" # shared by 'deepseek-ai' and 'Qwen'

BOS_TOKEN = {
    'base': "",

    'meta-llama': "<|begin_of_text|>",

    'deepseek-ai': "<｜begin▁of▁sentence｜>",

    'Qwen': "",
}

EOS_TOKEN = {
    'base': "",

    'meta-llama': "<|eot_id|>",

    'deepseek-ai': "<｜end▁of▁sentence｜>",

    'Qwen': "<|im_end|>\n",
}
