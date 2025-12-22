# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
env VLLM_PRECOMPILED_WHEEL_LOCATION="https://wheels.vllm.ai/dc9905368dd6f298395adaa20eeec37415c0cefe/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl" VLLM_USE_PRECOMPILED=1 uv pip install --editable . -v
"""  # noqa: E501

import os

os.environ["VLLM_LOGGING_LEVEL"] = "DEBUG"
os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"

from vllm import LLM, SamplingParams

PROMPTS = [
    "List 10 numbers only contains digit 1:",
]
TARGET = os.path.expanduser("~/huggingface/Qwen3-1.7B")
DRAFT = os.path.expanduser("~/huggingface/Qwen3-1.7B_eagle3")

NUM_SPEC_TOKENS = 5
TEMPERATURE = 0
OUTPUT_LEN = 32


def main():
    llm = LLM(
        model=TARGET,
        tensor_parallel_size=1,
        enable_chunked_prefill=False,
        enforce_eager=True,
        gpu_memory_utilization=0.5,
        speculative_config={
            "model": DRAFT,
            "num_speculative_tokens": NUM_SPEC_TOKENS,
            "method": "eagle3",
        },
        max_model_len=32,
        max_num_seqs=1,
        disable_log_stats=False,
    )

    sampling_params = SamplingParams(temperature=TEMPERATURE, max_tokens=OUTPUT_LEN)
    outputs = llm.generate(PROMPTS, sampling_params, use_tqdm=False)

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


if __name__ == "__main__":
    main()
