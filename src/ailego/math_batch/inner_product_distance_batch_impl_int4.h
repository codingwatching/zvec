// Copyright 2025-present the zvec project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

namespace zvec::ailego::DistanceBatch {

#if defined(__AVX2__)

static const __m256i MASK_INT4_AVX = _mm256_set1_epi32(0x0f0f0f0f);
static const AILEGO_ALIGNED(32) int8_t Int4ConvertTable[32] = {
    0, 1, 2, 3, 4, 5, 6, 7, -8, -7, -6, -5, -4, -3, -2, -1,
    0, 1, 2, 3, 4, 5, 6, 7, -8, -7, -6, -5, -4, -3, -2, -1};
static const __m256i INT4_LOOKUP_AVX =
    _mm256_load_si256((const __m256i *)Int4ConvertTable);
static const __m256i ONES_INT16_AVX = _mm256_set1_epi32(0x00010001);

template <size_t dp_batch>
static void compute_one_to_many_avx2_int4(
    const uint8_t *query, const uint8_t **ptrs,
    std::array<const uint8_t *, dp_batch> &prefetch_ptrs, size_t dimensionality,
    float *results) {
  dimensionality >>= 1;
  __m256i accs[dp_batch];
  for (size_t i = 0; i < dp_batch; ++i) {
    accs[i] = _mm256_setzero_si256();
  }
  size_t dim = 0;
  for (; dim + 32 <= dimensionality; dim += 32) {
    __m256i q = _mm256_loadu_si256((const __m256i *)(query + dim));
    __m256i q0 = _mm256_shuffle_epi8(INT4_LOOKUP_AVX,
                                     _mm256_and_si256(q, MASK_INT4_AVX));
    __m256i q1 = _mm256_shuffle_epi8(
        INT4_LOOKUP_AVX,
        _mm256_and_si256(_mm256_srli_epi16(q, 4), MASK_INT4_AVX));
    __m256i q0_abs = _mm256_abs_epi8(q0);
    __m256i q1_abs = _mm256_abs_epi8(q1);
    __m256i data_regs[dp_batch];
    for (size_t i = 0; i < dp_batch; ++i) {
      data_regs[i] = _mm256_loadu_si256((const __m256i *)(ptrs[i] + dim));
    }
    if (prefetch_ptrs[0]) {
      for (size_t i = 0; i < dp_batch; ++i) {
        ailego_prefetch(prefetch_ptrs[i] + dim);
      }
    }
    for (size_t i = 0; i < dp_batch; ++i) {
      __m256i data0 = _mm256_shuffle_epi8(
          INT4_LOOKUP_AVX, _mm256_and_si256(data_regs[i], MASK_INT4_AVX));
      __m256i data1 = _mm256_shuffle_epi8(
          INT4_LOOKUP_AVX,
          _mm256_and_si256(_mm256_srli_epi16(data_regs[i], 4), MASK_INT4_AVX));
      data0 = _mm256_sign_epi8(data0, q0);
      data1 = _mm256_sign_epi8(data1, q1);
      data0 = _mm256_madd_epi16(_mm256_maddubs_epi16(q0_abs, data0),
                                ONES_INT16_AVX);
      data1 = _mm256_madd_epi16(_mm256_maddubs_epi16(q1_abs, data1),
                                ONES_INT16_AVX);
      accs[i] = _mm256_add_epi32(_mm256_add_epi32(data0, data1), accs[i]);
    }
  }
  std::array<int, dp_batch> temp_results;
  for (size_t i = 0; i < dp_batch; ++i) {
    __m128i lo = _mm256_castsi256_si128(accs[i]);
    __m128i hi = _mm256_extracti128_si256(accs[i], 1);
    __m128i sum128 = _mm_add_epi32(lo, hi);
    sum128 = _mm_hadd_epi32(sum128, sum128);
    sum128 = _mm_hadd_epi32(sum128, sum128);
    temp_results[i] = _mm_cvtsi128_si32(sum128);
  }
  for (; dim < dimensionality; ++dim) {
    uint8_t q = query[dim];
    for (size_t i = 0; i < dp_batch; ++i) {
      uint8_t m = ptrs[i][dim];
      temp_results[i] +=
          Int4MulTable[(((m) << 4) & 0xf0) | (((q) >> 0) & 0xf)] +
          Int4MulTable[(((m) >> 0) & 0xf0) | (((q) >> 4) & 0xf)];
    }
  }
  for (size_t i = 0; i < dp_batch; ++i) {
    results[i] = static_cast<float>(temp_results[i]);
  }
}

#endif

}  // namespace zvec::ailego::DistanceBatch