/*******************************************************************************
 * include/wx_ppc.hpp
 *
 * Copyright (C) 2017 Florian Kurpicz <florian.kurpicz@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-2 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include <pwm/construction/building_blocks.hpp>
#include <pwm/construction/ctx_generic.hpp>
#include <pwm/construction/ppc.hpp>
#include <pwm/construction/wavelet_structure.hpp>
#include <pwm/util/common.hpp>

#include <pwm/wx_base.hpp>

template <typename AlphabetType, bool is_tree_>
class wx_ppc : public wx_in_out_external<false, false>  {

public:
  static constexpr bool is_parallel = true;
  static constexpr bool is_tree = is_tree_;
  static constexpr uint8_t word_width = sizeof(AlphabetType);
  static constexpr bool is_huffman_shaped = false;

  using ctx_t = ctx_generic<is_tree,
                            ctx_options::borders::sharded_single_level,
                            ctx_options::hist::all_level,
                            ctx_options::pre_computed_rho,
                            ctx_options::bv_initialized,
                            bit_vectors>;

  template <typename InputType>
  static wavelet_structure_tree
  compute(const InputType& text, const uint64_t size, const uint64_t levels) {

    if (size == 0) {
      return wavelet_structure_tree();
    }

    const auto rho = rho_dispatch<is_tree>::create(levels);
    ctx_t ctx(size, levels, rho, levels);

    ppc(text, size, levels, ctx);

    return wavelet_structure_tree(std::move(ctx.bv()));
  }
}; // class wx_ppc

/******************************************************************************/
