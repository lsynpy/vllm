# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os

from vllm import LLM, SamplingParams

path = os.path.expanduser("~/huggingface/TinyLlama-1.1B-Chat-v1.0-INT8/")
model = LLM(path)
sampling_params = SamplingParams(max_tokens=256)
outputs = model.generate("What is machine learning?", sampling_params)
for output in outputs:
    print(output.outputs[0].text)
