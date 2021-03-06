/*******************************************************************************
 * include/construction/ppc.hpp
 *
 * Copyright (C) 2017 Florian Kurpicz <florian.kurpicz@tu-dortmund.de>
 * Copyright (C) 2018 Marvin Löbel <loebel.marvin@gmail.com>
 *
 * All rights reserved. Published under the BSD-2 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include <omp.h>

#include <pwm/arrays/span.hpp>
#include <pwm/construction/building_blocks.hpp>
#include <pwm/construction/wavelet_structure.hpp>
#include <pwm/util/common.hpp>

template <typename AlphabetType, typename ContextType>
void ppc(AlphabetType const* text,
         const uint64_t size,
         const uint64_t levels,
         ContextType& ctx) {
  auto&& bv = ctx.bv();

  // compute initial histogram
  {
    const uint64_t alphabet_size = (1ull << levels);

    // Allocate a histogram buffer per thread
    helper_array sharded_hists(omp_get_max_threads(), alphabet_size);

    // Write the first level, and fill the histograms in parallel
    #pragma omp parallel
    {
      const auto shard = omp_get_thread_num();
      auto&& hist = sharded_hists[shard];

      omp_write_bits_wordwise(0, size, bv[0], [&](uint64_t const i) {
        auto const c = text[i];
        hist[c]++;
        uint64_t const bit = ((c >> (levels - 1)) & 1ULL);
        return bit;
      });
    }

    // Accumulate the histograms
    auto&& hist = ctx.hist_at_level(levels);
    #pragma omp parallel for
    for (uint64_t j = 0; j < alphabet_size; ++j) {
      for (uint64_t shard = 0; shard < sharded_hists.levels(); ++shard) {
        hist[j] += sharded_hists[shard][j];
      }
    }
  }

  auto&& leaf_hist = ctx.hist_at_level(levels);

  // Compute the histogram for each level of the wavelet structure
  #pragma omp parallel for schedule(nonmonotonic : dynamic, 1)
  for (uint64_t level = 1; level < levels; ++level) {
    auto&& hist = ctx.hist_at_level(level);
    const uint64_t blocks = (1 << level);
    const uint64_t required_characters = (1 << (levels - level));
    for (uint64_t i = 0; i < blocks; ++i) {
      for (uint64_t j = 0; j < required_characters; ++j) {
        hist[i] += leaf_hist[(i * required_characters) + j];
      }
    }
  }

  // Now we compute the wavelet structure bottom-up, i.e., the last level
  // first
  #pragma omp parallel for schedule(nonmonotonic : dynamic, 1)
  for (uint64_t level = 1; level < levels; ++level) {
    const uint64_t blocks = (1 << level);

    auto borders = ctx.borders_at_shard(level).slice(0, blocks);

    compute_borders_optional_zeros_rho(level, blocks, ctx, borders);

    // Now we insert the bits with respect to their bit prefixes
    for (uint64_t i = 0; i < size; ++i) {
      write_symbol_bit(bv, level, levels, borders, text[i]);
    }
  }

  ctx.hist_at_level(0)[0] = size;
  ctx.discard_borders();
}

/******************************************************************************/
