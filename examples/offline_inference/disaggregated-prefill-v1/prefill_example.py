# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig


def read_prompts():
    return [
        "Hello, my name is",
        "The capital of France is",
        "Your name is",
        "The capital of China is",
    ]


def main():
    prompts = read_prompts()

    sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=1)

    llm = LLM(
        model="Qwen/Qwen3-0.6B",
        enforce_eager=True,
        gpu_memory_utilization=0.8,
        max_num_batched_tokens=64,
        max_num_seqs=16,
        kv_transfer_config=KVTransferConfig(
            kv_connector="SharedStorageConnector",
            kv_role="kv_both",
            kv_connector_extra_config={"shared_storage_path": "local_storage"},
        ),
        max_model_len=2048,
    )

    # 1ST generation (prefill instance)
    outputs = llm.generate(
        prompts,
        sampling_params,
    )

    new_prompts = []
    print("-" * 30)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        new_prompts.append(prompt + generated_text)
        print(f"Prompt: {prompt!r}\nGenerated text: {generated_text!r}")
        print("-" * 30)

    # Write new_prompts to output.txt
    with open("output.txt", "w") as f:
        for prompt in new_prompts:
            f.write(prompt + "\n")
    print(f"Saved {len(new_prompts)} prompts to output.txt")


if __name__ == "__main__":
    main()
