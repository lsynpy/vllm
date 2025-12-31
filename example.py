# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os

os.environ["VLLM_LOGGING_LEVEL"] = "DEBUG"
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# os.environ["TRITON_INTERPRET"] = "1"

from vllm import LLM, SamplingParams

PROMPTS = [
    "List the first ten prime numbers:",
    # "The capital of France is",
    # "Once upon a time in a land far, far away,",
    # "List 10 numbers only contains digit 1:",
]

MODEL = os.path.expanduser("~/huggingface/Qwen3-1.7B")

TEMPERATURE = 0
OUTPUT_LEN = 8


def main():
    llm = LLM(
        model=MODEL,
        tensor_parallel_size=1,
        enable_chunked_prefill=False,
        enforce_eager=True,
        gpu_memory_utilization=0.5,
        max_model_len=32,
        max_num_seqs=4,
    )

    sampling_params = SamplingParams(temperature=TEMPERATURE, max_tokens=OUTPUT_LEN)
    outputs = llm.generate(PROMPTS, sampling_params, use_tqdm=False)

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


if __name__ == "__main__":
    main()
