# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import pytest

import tvm
import tvm.testing

import numpy as np


@tvm.testing.requires_package("transformers")
def test_whisper_preprocess_audio():
    from transformers.models.whisper.feature_extraction_whisper import WhisperFeatureExtractor

    samples = np.random.uniform(-0.5, 0.5, 480000).astype(np.float32)
    f = tvm.get_global_func("vm.builtin.whisper_process_audio")

    samples_nd = tvm.nd.array(samples)
    out = f(samples_nd)
    std_out = WhisperFeatureExtractor()._np_extract_fbank_features(samples)

    assert np.allclose(out.numpy(), std_out.T, atol=1e-4)


@tvm.testing.requires_package("transformers")
@pytest.mark.parametrize(
    "input_ids",
    [
        [50258],
        [
            50258,
            50259,
        ],
        [
            50258,
            50259,
            50359,
        ],
        [
            50258,
            50259,
            50359,
            50363,
        ],
        [
            50258,
            50259,
            50359,
            50363,
            2221,
        ],
        [
            50258,
            50259,
            50359,
            50363,
            2221,
            13,
        ],
    ],
)
def test_whisper_process_logits(input_ids):
    import torch
    from transformers.generation.logits_process import (
        LogitsProcessorList,
        SuppressTokensLogitsProcessor,
        SuppressTokensAtBeginLogitsProcessor,
        ForceTokensLogitsProcessor,
    )
    from transformers import WhisperConfig

    config = WhisperConfig.from_pretrained("openai/whisper-medium")
    processor_list = LogitsProcessorList()
    processor_list.append(SuppressTokensLogitsProcessor(config.suppress_tokens))
    processor_list.append(SuppressTokensAtBeginLogitsProcessor(config.begin_suppress_tokens, 4))
    processor_list.append(ForceTokensLogitsProcessor(config.forced_decoder_ids))

    input_ids = [input_ids]
    print(input_ids)
    next_token_logits = np.random.rand(1, 51865).astype(np.float32)

    nd_next_token_logits = tvm.nd.array(next_token_logits)
    f = tvm.get_global_func("vm.builtin.whisper_process_logits")
    f(nd_next_token_logits, len(input_ids[0]))

    pt_input_ids = torch.tensor(input_ids, dtype=torch.long)
    pt_next_token_logits = torch.from_numpy(next_token_logits)
    std_out = processor_list(pt_input_ids, pt_next_token_logits)

    a = nd_next_token_logits.numpy()
    b = std_out.numpy()
    # ignore -inf
    print(np.allclose(a[a > -3.4e38], b[b > -3.4e38]))


if __name__ == "__main__":
    tvm.testing.main()
