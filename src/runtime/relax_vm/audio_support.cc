/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
/*!
 * \file src/runtime/relax_vm/audio_support.cc
 * \brief Runtime to support ASR/TTS models
 */
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/relax_vm/vm.h>
#include <tvm/tir/expr.h>

#include <cmath>
#include <complex>

namespace tvm {
namespace runtime {
namespace relax_vm {

std::vector<double> hanning_window(int M, int window_size) {
  std::vector<double> window;
  window.resize(window_size);
  for (int i = 0; i < window_size; i++) {
    window[i] = 0.5 - 0.5 * std::cos(2 * M_PI * i / (M - 1));
  }

  return window;
}

std::vector<double> dft(const std::vector<double>& x) {
  int N = x.size();

  std::vector<double> result;
  result.resize(N * 2);

  for (int k = 0; k < N; k++) {
    for (int n = 0; n < N; n++) {
      std::complex<double> W = std::polar(1.0, -2 * M_PI / N * k * n);
      result[k * 2 + 0] += x[n] * W.real();
      result[k * 2 + 1] += x[n] * W.imag();
    }
  }
  return result;
}

std::vector<double> fft(const std::vector<double>& x) {
  int N = x.size();

  if (N % 2 == 1) {
    return dft(x);
  }

  std::vector<double> result;
  result.resize(N * 2);

  std::vector<double> even_x;
  std::vector<double> odd_x;

  even_x.reserve(N / 2);
  odd_x.reserve(N / 2);

  for (int i = 0; i < N; i++) {
    if (i % 2 == 0) {
      even_x.push_back(x[i]);
    } else {
      odd_x.push_back(x[i]);
    }
  }

  std::vector<double> E = fft(even_x);  // X_{0, ...., N/2 - 1}
  std::vector<double> O = fft(odd_x);   // X_{N / 2, ..., N - 1}

  for (int k = 0; k < N / 2; k++) {
    std::complex<double> p = {E[2 * k], E[2 * k + 1]};
    std::complex<double> q =
        std::polar(1.0, -2 * M_PI / N * k) * std::complex<double>(O[2 * k], O[2 * k + 1]);

    std::complex<double> e = p + q;
    std::complex<double> o = p - q;

    result[2 * k] = e.real();
    result[2 * k + 1] = e.imag();

    result[2 * (k + N / 2)] = o.real();
    result[2 * (k + N / 2) + 1] = o.imag();
  }

  return result;
}

std::vector<std::vector<double>> get_mel_filters(int sr, int n_fft, int n_mels) {
  std::vector<double> fftfreqs;
  double val = 1.0 / (n_fft * 1.0 / sr);
  int N = n_fft / 2 + 1;
  fftfreqs.resize(N);
  for (int i = 0; i < N; ++i) {
    fftfreqs[i] = i * val;
  }

  double min_mel = 0.0;
  double max_mel = 45.245640471924965;
  double melstep = (max_mel - min_mel) / (n_mels + 1.0);

  std::vector<double> mels;
  mels.resize(n_mels + 2);
  for (int i = 0; i < n_mels + 2; ++i) {
    mels[i] = min_mel + i * melstep;
  }

  double f_min = 0.0;
  double f_sp = 200.0 / 3;
  std::vector<double> freqs;
  freqs.resize(n_mels + 2);
  for (int i = 0; i < n_mels + 2; ++i) {
    freqs[i] = f_min + f_sp * mels[i];
  }

  double min_log_hz = 1000.0;
  double min_log_mel = (min_log_hz - f_min) / f_sp;
  double logstep = std::log(6.4) / 27.0;

  for (int i = 0; i < n_mels + 2; ++i) {
    if (mels[i] >= min_log_mel) {
      freqs[i] = min_log_hz * std::exp(logstep * (mels[i] - min_log_mel));
    }
  }

  std::vector<double> fdiff;
  fdiff.resize(n_mels + 1);
  for (int i = 0; i < n_mels + 1; ++i) {
    fdiff[i] = freqs[i + 1] - freqs[i];
  }

  std::vector<std::vector<double>> ramps;
  ramps.resize(freqs.size());

  for (size_t i = 0; i < freqs.size(); ++i) {
    ramps[i].resize(fftfreqs.size());
    for (size_t j = 0; j < fftfreqs.size(); ++j) {
      ramps[i][j] = freqs[i] - fftfreqs[j];
    }
  }

  std::vector<std::vector<double>> filters;
  filters.resize(ramps.size() - 2);
  for (size_t i = 0; i < ramps.size() - 2; i++) {
    filters[i].resize(ramps[0].size());
    double enorm = 2.0 / (freqs[i + 2] - freqs[i]);
    for (size_t j = 0; j < ramps[0].size(); ++j) {
      double lower = -ramps[i][j] / fdiff[i];
      double upper = ramps[i + 2][j] / fdiff[i + 1];
      filters[i][j] = enorm * std::max(0.0, std::min(lower, upper));
    }
  }
  return filters;
}

void log_mel_spec(const std::vector<double>& sample_data, int num_id,
                  const std::vector<double>& window, int n_fft, int hop_length,
                  const std::vector<std::vector<double>>& mel_filters, double* log_specs) {
  std::vector<double> frame;

  frame.resize(n_fft);
  for (int i = 0; i < n_fft; i++) {
    frame[i] = sample_data[num_id * hop_length + i] * window[i];
  }

  std::vector<double> fft_out = fft(frame);

  for (int i = 0; i < n_fft; i++) {
    fft_out[i] = fft_out[2 * i] * fft_out[2 * i] + fft_out[2 * i + 1] * fft_out[2 * i + 1];
  }

  for (int i = 0; i < n_fft / 2 + 1; i++) {
    fft_out[i] = 0.5 * (fft_out[i] + fft_out[n_fft - i]);
  }

  int n_mels = mel_filters.size();
  for (int i = 0; i < n_mels; i++) {
    double matmul_result = 0.0;
    for (int k = 0; k < n_fft / 2 + 1; k++) {
      matmul_result += fft_out[k] * mel_filters[i][k];
    }
    matmul_result = std::max(matmul_result, 1e-10);
    matmul_result = std::log10(matmul_result);
    log_specs[num_id * n_mels + i] = matmul_result;
  }
}

void WhisperProcessAudio(NDArray raw_speech, NDArray out_features) {
  ICHECK(raw_speech.IsContiguous());
  ICHECK(raw_speech.DataType() == DataType::Float(32)) << "raw speech data type is not float32!";
  ICHECK(raw_speech->device.device_type == kDLCPU) << "raw speech device must be CPU!";
  ICHECK_EQ(raw_speech->ndim, 1);

  const float* p_data = static_cast<float*>(raw_speech->data);
  float* p_out = static_cast<float*>(out_features->data);

  int sampling_rate = 16000;
  int n_fft = 400;
  int n_mels = 80;
  int max_length = 480000;
  int hop_length = 160;

  std::vector<std::vector<double>> mel_filters = get_mel_filters(sampling_rate, n_fft, n_mels);

  std::vector<double> window = hanning_window(n_fft + 1, n_fft);

  std::vector<double> pad_data;
  pad_data.resize(max_length + n_fft);
  for (int i = 0; i < n_fft / 2; ++i) {
    pad_data[n_fft / 2 - 1 - i] = p_data[i + 1];
  }
  for (int i = 0; i < max_length; ++i) {
    if (i >= raw_speech->shape[0]) {
      pad_data[n_fft / 2 + i] = 0.0;
    } else {
      pad_data[n_fft / 2 + i] = p_data[i];
    }
  }
  for (int i = 0; i < n_fft / 2; ++i) {
    pad_data[n_fft / 2 + max_length + i] = pad_data[n_fft / 2 + max_length - 2 - i];
  }

  int num_frames = 1 + (pad_data.size() - n_fft) / hop_length;
  double* log_specs;
  int output_num_frames = (num_frames - 1);
  log_specs = new double[output_num_frames * n_mels];
  for (int i = 0; i < output_num_frames; ++i) {
    log_mel_spec(pad_data, i, window, n_fft, hop_length, mel_filters, log_specs);
  }

  double log_specs_max = std::numeric_limits<float>::min();
  for (int i = 0; i < output_num_frames; i++) {
    for (int j = 0; j < n_mels; ++j) {
      log_specs_max = std::max(log_specs_max, log_specs[i * n_mels + j]);
    }
  }

  log_specs_max -= 8.0;
  for (int i = 0; i < output_num_frames; i++) {
    for (int j = 0; j < n_mels; ++j) {
      log_specs[i * n_mels + j] = std::max(log_specs_max, log_specs[i * n_mels + j]);
      log_specs[i * n_mels + j] = (log_specs[i * n_mels + j] + 4.0) / 4.0;
    }
  }

  for (int i = 0; i < output_num_frames; i++) {
    for (int j = 0; j < n_mels; ++j) {
      p_out[j * output_num_frames + i] = static_cast<float>(log_specs[i * n_mels + j]);
    }
  }
}

TVM_REGISTER_GLOBAL("vm.builtin.whisper_process_audio").set_body_typed(WhisperProcessAudio);

// This is an inplace operation.
void WhisperProcessLogits(NDArray logits, double cur_len, Array<Integer> suppress_tokens,
                          Array<Integer> begin_suppress_tokens,
                          Array<Array<Integer>> forced_decoder_ids) {
  std::unordered_map<int, int> forced_decoder_ids_map = {};
  for (size_t i = 0; i < forced_decoder_ids.size(); i++) {
    forced_decoder_ids_map[forced_decoder_ids[i][0].IntValue()] =
        forced_decoder_ids[i][1].IntValue();
  }
  ICHECK(logits.IsContiguous());
  ICHECK(logits.DataType() == DataType::Float(32)) << "Logits data type is not float32!";
  ICHECK(logits->device.device_type == kDLCPU) << "logits device must be CPU!";

  float* logits_raw_data = static_cast<float*>(logits->data);

  float neg_inf = std::numeric_limits<float>::lowest();

  for (size_t i = 0; i < suppress_tokens.size(); ++i) {
    logits_raw_data[suppress_tokens[i].IntValue()] = neg_inf;
  }

  int generated_length = cur_len;
  if (generated_length == 4) {
    for (size_t i = 0; i < begin_suppress_tokens.size(); ++i) {
      logits_raw_data[begin_suppress_tokens[i].IntValue()] = neg_inf;
    }
  }

  if (forced_decoder_ids_map.count(generated_length) > 0) {
    int current_token = forced_decoder_ids_map[generated_length];
    for (auto i = 0; i < logits->shape[logits->ndim - 1]; ++i) {
      if (i == current_token) {
        logits_raw_data[i] = 0;
      } else {
        logits_raw_data[i] = neg_inf;
      }
    }
  }
}

TVM_REGISTER_GLOBAL("vm.builtin.whisper_process_logits").set_body_typed(WhisperProcessLogits);

}  // namespace relax_vm
}  // namespace runtime
}  // namespace tvm