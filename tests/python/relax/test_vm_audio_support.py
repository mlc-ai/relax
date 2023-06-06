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


if __name__ == "__main__":
    tvm.testing.main()
