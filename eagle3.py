# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os

from vllm import LLM, SamplingParams

PROMPTS = [
    "List 10 numbers only contains digit 1:",
]
TARGET = os.path.expanduser("~/huggingface/Qwen3-1.7B")
DRAFT = os.path.expanduser("~/huggingface/Qwen3-1.7B_eagle3")

sampling_params = SamplingParams(temperature=0.0, max_tokens=32)

llm = LLM(
    model=TARGET,
    tensor_parallel_size=1,
    enable_chunked_prefill=False,
    enforce_eager=True,
    gpu_memory_utilization=0.7,
    speculative_config={
        "model": DRAFT,
        "draft_tensor_parallel_size": 1,
        "num_speculative_tokens": 2,
        "method": "eagle3",
    },
    max_model_len=32,
    max_num_seqs=1,
)

outputs = llm.generate(PROMPTS, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
