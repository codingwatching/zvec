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

#include <vector>
#include <ailego/internal/cpu_features.h>
#include <ailego/utility/math_helper.h>
#include <zvec/ailego/internal/platform.h>
#include <zvec/ailego/math_batch/utils.h>
#include <zvec/ailego/utility/type_helper.h>
#include "euclidean_distance_batch_impl.h"
#include "euclidean_distance_batch_impl_fp16.h"
#include "euclidean_distance_batch_impl_int8.h"

namespace zvec::ailego::DistanceBatch {

//SquaredEuclideanDistanceBatch 
template <typename T, size_t BatchSize, size_t PrefetchStep, typename = void>
struct SquaredEuclideanDistanceBatch;

// Function template partial specialization is not allowed,
// therefore the wrapper struct is required.
template <typename T, size_t BatchSize>
struct SquaredEuclideanDistanceBatchImpl {
  using ValueType = typename std::remove_cv<T>::type;
  static void compute_one_to_many(
      const ValueType *query, const ValueType **ptrs,
      std::array<const ValueType *, BatchSize> &prefetch_ptrs, size_t dim,
      float *sums) {
    return compute_one_to_many_squared_euclidean_fallback(query, ptrs, prefetch_ptrs, dim, sums);
  }
};

template <size_t BatchSize>
struct SquaredEuclideanDistanceBatchImpl<float, BatchSize> {
  using ValueType = float;
  static void compute_one_to_many(
      const ValueType *query, const ValueType **ptrs,
      std::array<const ValueType *, BatchSize> &prefetch_ptrs, size_t dim,
      float *sums) {
#if defined(__AVX512F__)
    if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX2) {
      return compute_one_to_many_squared_euclidean_avx512f_fp32<ValueType, BatchSize>(
          query, ptrs, prefetch_ptrs, dim, sums);
    }
#endif

#if defined(__AVX2__)
    if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX2) {
      return compute_one_to_many_squared_euclidean_avx2_fp32<ValueType, BatchSize>(
          query, ptrs, prefetch_ptrs, dim, sums);
    }
#endif
    return compute_one_to_many_squared_euclidean_fallback(query, ptrs, prefetch_ptrs, dim, sums);
  }
};

template <size_t BatchSize>
struct SquaredEuclideanDistanceBatchImpl<int8_t, BatchSize> {
  using ValueType = int8_t;
  static void compute_one_to_many(
      const int8_t *query, const int8_t **ptrs,
      std::array<const int8_t *, BatchSize> &prefetch_ptrs, size_t dim,
      float *sums) {
#if defined(__AVX2__)
    if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX2) {
      return compute_one_to_many_squared_euclidean_avx2_int8<ValueType, BatchSize>(
          query, ptrs, prefetch_ptrs, dim, sums);
    }
#endif
    return compute_one_to_many_squared_euclidean_fallback(query, ptrs, prefetch_ptrs, dim, sums);
  }
};

template <size_t BatchSize>
struct SquaredEuclideanDistanceBatchImpl<ailego::Float16, BatchSize> {
  using ValueType = ailego::Float16;
  static void compute_one_to_many(
      const ailego::Float16 *query, const ailego::Float16 **ptrs,
      std::array<const ailego::Float16 *, BatchSize> &prefetch_ptrs, size_t dim,
      float *sums) {
#if defined(__AVX512FP16__)
    if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512_FP16) {
      return compute_one_to_many_squared_euclidean_avx512fp16_fp16<ValueType, BatchSize>(
          query, ptrs, prefetch_ptrs, dim, sums);
    }
#endif
#if defined(__AVX512F__)
    if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
      return compute_one_to_many_squared_euclidean_avx512f_fp16<ValueType, BatchSize>(
          query, ptrs, prefetch_ptrs, dim, sums);
    }
#endif
#if defined(__AVX2__)
    if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX2) {
      return compute_one_to_many_squared_euclidean_avx2_fp16<ValueType, BatchSize>(
          query, ptrs, prefetch_ptrs, dim, sums);
    }
#endif
    return compute_one_to_many_squared_euclidean_fallback(query, ptrs, prefetch_ptrs, dim, sums);
  }
};

template <typename T, size_t BatchSize, size_t PrefetchStep, typename>
struct SquaredEuclideanDistanceBatch {
  using ValueType = typename std::remove_cv<T>::type;

  static inline void ComputeBatch(const ValueType **vecs,
                                  const ValueType *query, size_t num_vecs,
                                  size_t dim, float *results) {
    size_t i = 0;
    for (; i + BatchSize <= num_vecs; i += BatchSize) {
      std::array<const ValueType *, BatchSize> prefetch_ptrs;
      for (size_t j = 0; j < BatchSize; ++j) {
        if (i + j + BatchSize * PrefetchStep < num_vecs) {
          prefetch_ptrs[j] = vecs[i + j + BatchSize * PrefetchStep];
        } else {
          prefetch_ptrs[j] = nullptr;
        }
      }
      SquaredEuclideanDistanceBatchImpl<ValueType, BatchSize>::compute_one_to_many(
          query, &vecs[i], prefetch_ptrs, dim, &results[i]);
    }
    for (; i < num_vecs; ++i) {  // TODO: unroll by 1, 2, 4, 8, etc.
      std::array<const ValueType *, 1> prefetch_ptrs{nullptr};
      SquaredEuclideanDistanceBatchImpl<ValueType, 1>::compute_one_to_many(
          query, &vecs[i], prefetch_ptrs, dim, &results[i]);
    }
  }
};

//EuclideanDistanceBatch 
template <typename T, size_t BatchSize, size_t PrefetchStep, typename = void>
struct EuclideanDistanceBatch;

template <typename T, size_t BatchSize, size_t PrefetchStep, typename>
struct EuclideanDistanceBatch {
  using ValueType = typename std::remove_cv<T>::type;

  static inline void ComputeBatch(const ValueType **vecs,
                                  const ValueType *query, size_t num_vecs,
                                  size_t dim, float *results) {
    SquaredEuclideanDistanceBatch<T, BatchSize, PrefetchStep>::ComputeBatch(vecs, query, num_vecs, dim, results);

    for (size_t i=0; i<num_vecs; ++i) {
      results[i] = std::sqrt(results[i]);
    }
  }
};

}  // namespace zvec::ailego::DistanceBatch
