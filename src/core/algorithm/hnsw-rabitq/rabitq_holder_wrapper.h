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

#include "zvec/core/framework/index_provider.h"

namespace zvec {
namespace core {

/*! RaBitQ Holder Wrapper
 * Wraps quantized holder and provides reference to original data provider
 */
class RabitqHolderWrapper : public IndexProvider {
 public:
  //! Constructor
  RabitqHolderWrapper(IndexHolder::Pointer quantized,
                      IndexProvider::Pointer reference)
      : quantized_(quantized), reference_(reference) {}

  //! Destructor
  ~RabitqHolderWrapper() override = default;

  //! Create a new iterator - forward to quantized holder
  Iterator::Pointer create_iterator(void) override {
    return quantized_->create_iterator();
  }

  //! Retrieve count of vectors
  size_t count(void) const override {
    return quantized_->count();
  }

  //! Retrieve dimension of vector
  size_t dimension(void) const override {
    return quantized_->dimension();
  }

  //! Retrieve type of vector
  IndexMeta::DataType data_type(void) const override {
    return quantized_->data_type();
  }

  //! Retrieve vector size in bytes
  size_t element_size(void) const override {
    return quantized_->element_size();
  }

  //! Retrieve if it can multi-pass
  bool multipass(void) const override {
    return quantized_->multipass();
  }

  //! Retrieve a vector using a primary key - return quantized data
  const void *get_vector(uint64_t key) const override {
    auto provider = std::dynamic_pointer_cast<IndexProvider>(quantized_);
    if (provider) {
      return provider->get_vector(key);
    }
    return nullptr;
  }

  //! Retrieve a vector using a primary key with memory block
  int get_vector(const uint64_t key,
                 IndexStorage::MemoryBlock &block) const override {
    auto provider = std::dynamic_pointer_cast<IndexProvider>(quantized_);
    if (provider) {
      return provider->get_vector(key, block);
    }
    return IndexError_NotImplemented;
  }

  //! Retrieve the owner class
  const std::string &owner_class(void) const override {
    static const std::string owner = "RabitqHolderWrapper";
    return owner;
  }

 private:
  IndexHolder::Pointer quantized_;    // Quantized data
  IndexProvider::Pointer reference_;  // Original data
};

}  // namespace core
}  // namespace zvec
