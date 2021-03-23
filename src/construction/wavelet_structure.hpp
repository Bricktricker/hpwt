/*******************************************************************************
 * include/util/common.hpp
 *
 * Copyright (C) 2017 Marvin Löbel <loebel.marvin@gmail.com>
 *
 * All rights reserved. Published under the BSD-2 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include <memory>
#include <type_traits>
#include <typeinfo>
#include <vector>

#include "../arrays/bit_vectors.hpp"
//#include "huffman/huff_codes.hpp"
#include "../util/debug_assert.hpp"

class wavelet_structure;

class base_wavelet_structure {
  friend class wavelet_structure;

public:
  // Prevent accidental copies
  base_wavelet_structure(base_wavelet_structure const&) = delete;
  base_wavelet_structure& operator=(base_wavelet_structure const&) = delete;

  // Allow moving
  base_wavelet_structure(base_wavelet_structure&& other) = default;
  base_wavelet_structure& operator=(base_wavelet_structure&& other) = default;

  virtual ~base_wavelet_structure() = default;

  inline base_bit_vectors const& bvs() const {
    return bvs_;
  }

private:
  base_bit_vectors bvs_;
  size_t text_size_;
  bool is_tree_;
  bool is_huffman_shaped_;

protected:
  base_wavelet_structure(base_bit_vectors&& bvs,
                         bool is_tree,
                         bool is_huffman_shaped)
      : bvs_(std::move(bvs)),
        is_tree_(is_tree),
        is_huffman_shaped_(is_huffman_shaped) {
    if (bvs_.levels() > 0) {
      text_size_ = bvs_.level_bit_size(0);
    } else {
      text_size_ = 0;
    }
  }

  inline virtual std::vector<uint64_t> const& zeros() const {
    static std::vector<uint64_t> empty;
    return empty;
  }
  inline virtual void const* codes(std::type_info const&) const {
    DCHECK(false);
    return nullptr;
  }
};
class wavelet_structure_tree : public base_wavelet_structure {
public:
  wavelet_structure_tree() : wavelet_structure_tree(base_bit_vectors()) {}
  wavelet_structure_tree(base_bit_vectors&& bvs)
      : base_wavelet_structure(std::move(bvs), true, false) {}
};

class wavelet_structure_matrix : public base_wavelet_structure {
public:
  wavelet_structure_matrix()
      : wavelet_structure_matrix(base_bit_vectors(), {}) {}
  wavelet_structure_matrix(base_bit_vectors&& bvs,
                           std::vector<uint64_t>&& zeros)
      : base_wavelet_structure(std::move(bvs), false, false),
        zeros_(std::move(zeros)) {}

private:
  std::vector<uint64_t> zeros_;
  inline virtual std::vector<uint64_t> const& zeros() const override {
    return zeros_;
  }
};

/******************************************************************************/
