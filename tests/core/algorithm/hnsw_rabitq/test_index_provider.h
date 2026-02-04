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

#include <unordered_map>
#include <vector>
#include "zvec/core/framework/index_holder.h"
#include "zvec/core/framework/index_meta.h"
#include "zvec/core/framework/index_provider.h"

namespace zvec::core {

class FloatIndexProvider : public IndexProvider {
 public:
  //! Index Holder Iterator
  class Iterator : public IndexHolder::Iterator {
   public:
    //! Constructor
    explicit Iterator(FloatIndexProvider *provider)
        : provider_(provider), current_idx_(0) {}

    //! Destructor
    virtual ~Iterator() = default;

    //! Retrieve pointer of data
    const void *data() const override {
      if (!is_valid()) {
        return nullptr;
      }
      return provider_->features_[current_idx_].second.data();
    }

    //! Test if the iterator is valid
    bool is_valid() const override {
      return current_idx_ < provider_->features_.size();
    }

    //! Retrieve primary key
    uint64_t key() const override {
      if (!is_valid()) {
        return 0;
      }
      return provider_->features_[current_idx_].first;
    }

    //! Next iterator
    void next() override {
      if (is_valid()) {
        ++current_idx_;
      }
    }

   private:
    FloatIndexProvider *provider_;
    size_t current_idx_;
  };

  //! Constructor
  explicit FloatIndexProvider(size_t dimension)
      : dimension_(dimension), owner_class_("FloatIndexProvider") {}

  //! Destructor
  virtual ~FloatIndexProvider() = default;

 public:  // IndexHolder interface implementation
  //! Retrieve count of elements in holder
  size_t count() const override {
    return features_.size();
  }

  //! Retrieve dimension
  size_t dimension() const override {
    return dimension_;
  }

  //! Retrieve type information
  IndexMeta::DataType data_type() const override {
    return IndexMeta::DataType::DT_FP32;
  }

  //! Retrieve element size in bytes
  size_t element_size() const override {
    return dimension_ * sizeof(float);
  }

  //! Retrieve if it can multi-pass
  bool multipass() const override {
    return true;
  }

  //! Create a new iterator
  IndexHolder::Iterator::Pointer create_iterator() override {
    return IndexHolder::Iterator::Pointer(new Iterator(this));
  }

 public:  // IndexProvider interface implementation
  //! Retrieve a vector using a primary key
  const void *get_vector(uint64_t key) const override {
    auto it = key_to_idx_map_.find(key);
    if (it == key_to_idx_map_.end()) {
      return nullptr;
    }
    return features_[it->second].second.data();
  }

  //! Retrieve the owner class
  const std::string &owner_class() const override {
    return owner_class_;
  }

 public:  // Helper methods
  //! Append an element into holder
  bool emplace(uint64_t key, const ailego::NumericalVector<float> &vec) {
    if (vec.size() != dimension_) {
      return false;
    }
    size_t idx = features_.size();
    features_.emplace_back(key, vec);
    key_to_idx_map_[key] = idx;
    return true;
  }

  //! Append an element into holder
  bool emplace(uint64_t key, ailego::NumericalVector<float> &&vec) {
    if (vec.size() != dimension_) {
      return false;
    }
    size_t idx = features_.size();
    features_.emplace_back(key, std::move(vec));
    key_to_idx_map_[key] = idx;
    return true;
  }

 private:
  size_t dimension_;
  std::unordered_map<uint64_t, size_t> key_to_idx_map_;
  std::vector<std::pair<uint64_t, ailego::NumericalVector<float>>> features_;
  std::string owner_class_;
};

}  // namespace zvec::core