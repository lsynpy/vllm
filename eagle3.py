# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os

import torch

os.environ["VLLM_LOGGING_LEVEL"] = "DEBUG"
os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# os.environ["TRITON_INTERPRET"] = "1"

from vllm import LLM, SamplingParams
from vllm.v1.metrics.reader import Counter, Vector

# Determinism
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
torch.use_deterministic_algorithms(True)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


PROMPTS = [
    "List the first ten prime numbers:",
    # "The capital of France is",
    # "Once upon a time in a land far, far away,",
    # "List 10 numbers only contains digit 1:",
]
TARGET = os.path.expanduser("~/huggingface/Qwen3-1.7B")
DRAFT = os.path.expanduser("~/huggingface/Qwen3-1.7B_eagle3")

NUM_SPEC_TOKENS = 3
TEMPERATURE = 0
OUTPUT_LEN = 8


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
        max_num_seqs=4,
        disable_log_stats=False,
    )

    sampling_params = SamplingParams(temperature=TEMPERATURE, max_tokens=OUTPUT_LEN)
    outputs = llm.generate(PROMPTS, sampling_params, use_tqdm=False)

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

    # Collect and print metrics
    metrics = llm.get_metrics()

    total_num_output_tokens = sum(
        len(output.outputs[0].token_ids) for output in outputs
    )
    num_drafts = 0
    num_draft_tokens = 0
    num_accepted_tokens = 0
    acceptance_counts = [0] * NUM_SPEC_TOKENS

    for metric in metrics:
        if metric.name == "vllm:spec_decode_num_drafts":
            assert isinstance(metric, Counter)
            num_drafts += metric.value
        elif metric.name == "vllm:spec_decode_num_draft_tokens":
            assert isinstance(metric, Counter)
            num_draft_tokens += metric.value
        elif metric.name == "vllm:spec_decode_num_accepted_tokens":
            assert isinstance(metric, Counter)
            num_accepted_tokens += metric.value
        elif metric.name == "vllm:spec_decode_num_accepted_tokens_per_pos":
            assert isinstance(metric, Vector)
            for pos in range(len(metric.values)):
                acceptance_counts[pos] += metric.values[pos]

    acceptance_length = 1 + (num_accepted_tokens / num_drafts) if num_drafts > 0 else 1

    print("-" * 50)
    print(f"total_num_output_tokens: {total_num_output_tokens}")
    print(f"num_drafts: {num_drafts}")
    print(f"num_draft_tokens: {num_draft_tokens}")
    print(f"num_accepted_tokens: {num_accepted_tokens}")
    print(f"mean acceptance length: {acceptance_length:.2f}")
    print("-" * 50)

    # Print acceptance at each token position
    for i in range(len(acceptance_counts)):
        acceptance_rate = acceptance_counts[i] / num_drafts if num_drafts > 0 else 0
        print(f"acceptance at token {i}: {acceptance_rate:.2f}")


if __name__ == "__main__":
    main()
