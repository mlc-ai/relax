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

#include <cmath>

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

std::vector<double> dft(const std::vector<double>& in) {
  int N = in.size();

  std::vector<double> out;
  out.resize(N * 2);

  for (int k = 0; k < N; k++) {
    double re = 0;
    double im = 0;

    for (int n = 0; n < N; n++) {
      double angle = 2 * M_PI * k * n / N;
      re += in[n] * std::cos(angle);
      im -= in[n] * std::sin(angle);
    }

    out[k * 2 + 0] = re;
    out[k * 2 + 1] = im;
  }
  return out;
}

std::vector<double> fft(const std::vector<double>& in) {
  std::vector<double> out;
  out.resize(in.size() * 2);

  int N = in.size();

  if (N == 1) {
    out[0] = in[0];
    out[1] = 0;
    return out;
  }

  if (N % 2 == 1) {
    return dft(in);
  }

  std::vector<double> even;
  std::vector<double> odd;

  even.reserve(N / 2);
  odd.reserve(N / 2);

  for (int i = 0; i < N; i++) {
    if (i % 2 == 0) {
      even.push_back(in[i]);
    } else {
      odd.push_back(in[i]);
    }
  }

  std::vector<double> even_fft = fft(even);
  std::vector<double> odd_fft = fft(odd);

  for (int k = 0; k < N / 2; k++) {
    double theta = 2 * M_PI * k / N;

    double re = std::cos(theta);
    double im = -std::sin(theta);

    double re_odd = odd_fft[2 * k + 0];
    double im_odd = odd_fft[2 * k + 1];

    out[2 * k + 0] = even_fft[2 * k + 0] + re * re_odd - im * im_odd;
    out[2 * k + 1] = even_fft[2 * k + 1] + re * im_odd + im * re_odd;

    out[2 * (k + N / 2) + 0] = even_fft[2 * k + 0] - re * re_odd + im * im_odd;
    out[2 * (k + N / 2) + 1] = even_fft[2 * k + 1] - re * im_odd - im * re_odd;
  }

  return out;
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
void WhisperProcessLogits(NDArray logits, double cur_len) {
  std::vector<int> suppress_tokens = {
      1,     2,     7,     8,     9,     10,    14,    25,    26,    27,    28,    29,    31,
      58,    59,    60,    61,    62,    63,    90,    91,    92,    93,    359,   503,   522,
      542,   873,   893,   902,   918,   922,   931,   1350,  1853,  1982,  2460,  2627,  3246,
      3253,  3268,  3536,  3846,  3961,  4183,  4667,  6585,  6647,  7273,  9061,  9383,  10428,
      10929, 11938, 12033, 12331, 12562, 13793, 14157, 14635, 15265, 15618, 16553, 16604, 18362,
      18956, 20075, 21675, 22520, 26130, 26161, 26435, 28279, 29464, 31650, 32302, 32470, 36865,
      42863, 47425, 49870, 50254, 50258, 50358, 50359, 50360, 50361, 50362};
  std::vector<int> begin_suppress_tokens = {220, 50257};
  std::unordered_map<int, int> forced_decoder_ids = {{1, 50259}, {2, 50359}, {3, 50363}};
  ICHECK(logits.IsContiguous());
  ICHECK(logits.DataType() == DataType::Float(32)) << "Logits data type is not float32!";
  ICHECK(logits->device.device_type == kDLCPU) << "logits device must be CPU!";

  float* logits_raw_data = static_cast<float*>(logits->data);

  float neg_inf = std::numeric_limits<float>::lowest();

  for (size_t i = 0; i < suppress_tokens.size(); ++i) {
    logits_raw_data[suppress_tokens[i]] = neg_inf;
  }

  int generated_length = cur_len;
  if (generated_length == 4) {
    for (size_t i = 0; i < begin_suppress_tokens.size(); ++i) {
      logits_raw_data[begin_suppress_tokens[i]] = neg_inf;
    }
  }

  if (forced_decoder_ids.count(generated_length) > 0) {
    int current_token = forced_decoder_ids[generated_length];
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
