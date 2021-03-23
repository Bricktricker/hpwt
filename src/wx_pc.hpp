/*******************************************************************************
 * include/wx_pc.hpp
 *
 * Copyright (C) 2017 Florian Kurpicz <florian.kurpicz@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-2 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include <vector>

#include "construction/ctx_generic.hpp"
#include "construction/pc.hpp"
#include "construction/wavelet_structure.hpp"

template <bool external_in_,
          bool external_out_,
          bool stats_support_ = false,
          bool is_inplace_ = false>
class wx_in_out_external {
public:
  static constexpr bool external_in = external_in_;
  static constexpr bool external_out = external_out_;
  static constexpr bool stats_support = stats_support_;
  static constexpr bool is_inplace = is_inplace_;
}; // class wx_in_out_external

template <typename AlphabetType, bool is_tree_>
class wx_pc : public wx_in_out_external<false, false>  {

public:
  static constexpr bool is_parallel = false;
  static constexpr bool is_tree = is_tree_;
  static constexpr uint8_t word_width = sizeof(AlphabetType);
  static constexpr bool is_huffman_shaped = false;

  using ctx_t = ctx_generic<is_tree,
                            ctx_options::borders::single_level,
                            ctx_options::hist::single_level,
                            ctx_options::live_computed_rho,
                            ctx_options::bv_initialized,
                            bit_vectors>;

  template <typename InputType>
  static wavelet_structure_tree compute(const InputType* text,
                                   const uint64_t size,
                                   const uint64_t levels) {

    if (size == 0) {
      return wavelet_structure_tree();
    }

    auto ctx = ctx_t(size, levels, levels);

    pc(text, size, levels, ctx);

    return wavelet_structure_tree(std::move(ctx.bv()));
  }
}; // class wx_pc

/******************************************************************************/
