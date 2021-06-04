#pragma once

#include <distwt/common/effective_alphabet.hpp>
#include <distwt/common/wt.hpp>
#include <pwm/construction/ctx_generic.hpp>
#include <pwm/construction/building_blocks.hpp>
#include <pwm/util/common.hpp>
#include <pwm/arrays/bit_vectors.hpp>

#include <omp.h>
#include <src/omp_write_bits.hpp>

// pps from the distwt repositiory, include/construction/pps.hpp
template <typename AlphabetType, typename ContextType>
void pps(AlphabetType const* text,
         const uint64_t size,
         const uint64_t levels,
         ContextType& ctx,
         wt_bits_t& bv) {
  auto sorted_text_ = std::vector<AlphabetType>(size);
  auto sorted_text = span<AlphabetType>(sorted_text_);

  std::vector<uint64_t> offsets_(1 << levels, 0);
  auto offsets = span<uint64_t>(offsets_);
  bv[0].resize(size);
  auto&& zeros = ctx.zeros();

  #pragma omp parallel
  {
    const auto omp_rank = omp_get_thread_num();
    const auto omp_size = omp_get_num_threads();
    const uint64_t alphabet_size = (1 << levels);

    {
      auto&& rank_hist = ctx.hist_at_shard(omp_rank);

      // While initializing the histogram, we also compute the first level
      omp_write_bits_vec(0, size, bv[0], [&](uint64_t const i) {
        rank_hist[text[i]]++;
        uint64_t bit = ((text[i] >> (levels - 1)) & 1ULL);
        return bit;
      });
    }

    #pragma omp single
    {
      if constexpr (ContextType::compute_zeros) {
        for (int32_t rank = 0; rank < omp_size; ++rank) {
          auto&& rank_hist = ctx.hist_at_shard(rank);
          compute_last_level_zeros(levels, zeros, rank_hist);
        }
      }
    }

    // Now we compute the wavelet structure bottom-up, i.e., the last level
    // first
    for (uint64_t level = levels - 1; level > 0; --level) {
      const uint64_t prefix_shift = (levels - level);
      const uint64_t cur_bit_shift = prefix_shift - 1;

      // Compute the histogram and the border for each bit prefix and
      // processor, i.e., for one fixed bit prefix we compute the prefix sum
      // over the number of occurrences at each processor
      #pragma omp for
      for (uint64_t i = 0; i < alphabet_size; i += (1ULL << prefix_shift)) {
        {
          auto&& hist = ctx.hist_at_shard(0);
          auto&& borders = ctx.borders_at_shard(0);

          hist[i] += hist[i + (1ULL << cur_bit_shift)];
          borders[i] = 0;
        }
        for (int32_t rank = 1; rank < omp_size; ++rank) {
          auto&& hist = ctx.hist_at_shard(rank);
          auto&& borders = ctx.borders_at_shard(rank);
          auto&& prev_borders = ctx.borders_at_shard(rank - 1);
          auto&& prev_hist = ctx.hist_at_shard(rank - 1);

          hist[i] += hist[i + (1ULL << cur_bit_shift)];
          borders[i] = prev_borders[i] + prev_hist[i];
        }
      }

      // Now we compute the offset for each bit prefix, i.e., the number of
      // lexicographically smaller characters
      #pragma omp single
      {        
        auto&& last_hist = ctx.hist_at_shard(omp_size - 1);

        //resize bit vectors
        const size_t num_level_nodes = (1ULL << level);
        const size_t glob_offs = (1ULL << level) - 1;
        const size_t multFac = 1ULL << (levels - level);
        for(size_t v = 0; v < num_level_nodes; v++) {
            size_t size = 0;
            for(int32_t shard = 0; shard < omp_size; shard++) {
              size += ctx.hist_at_shard(shard)[multFac * v];
            }
            const size_t node = glob_offs + v;
            bv[node].resize(size);
        }

        auto&& last_borders = ctx.borders_at_shard(omp_size - 1);
        for (uint64_t i = 1; i < (1ULL << level); ++i) {
          const auto rho = ctx.rho(level, i);
          const auto prev_rho = ctx.rho(level, i - 1);

          offsets[rho << prefix_shift] =
              offsets[prev_rho << prefix_shift] +
              last_borders[prev_rho << prefix_shift] +
              last_hist[prev_rho << prefix_shift];
          if (ContextType::compute_rho) {
            ctx.set_rho(level, i - 1, prev_rho >> 1);
          }
        }
        // The number of 0s is the position of the first 1 at the first
        // processor
        if constexpr (ContextType::compute_zeros) {
          zeros[level - 1] = offsets[1ULL << prefix_shift];
        }
      }
      // We add the offset to the borders (for performance)
      #pragma omp for
      for (int32_t rank = 0; rank < omp_size; ++rank) {
        auto&& borders = ctx.borders_at_shard(rank);
        for (uint64_t i = 0; i < alphabet_size; i += (1ULL << prefix_shift)) {
          borders[i] += offsets[i];
        }
      }

      // We align the borders (in memory) to increase performance by reducing
      // the number of cache misses
      std::vector<uint64_t> borders_aligned_(1ULL << level, 0);
      span<uint64_t> borders_aligned(borders_aligned_);
      {
        auto&& borders = ctx.borders_at_shard(omp_rank);
        for (uint64_t i = 0; i < alphabet_size; i += (1ULL << prefix_shift)) {
          borders_aligned[i >> prefix_shift] = borders[i];
        }
      }

      // Sort the text using the computed (and aligned) borders
      #pragma omp for
      for (uint64_t i = 0; i <= size - 64; i += 64) {
        for (uint64_t j = 0; j < 64; ++j) {
          const AlphabetType considerd_char = (text[i + j] >> cur_bit_shift);
          sorted_text[borders_aligned[considerd_char >> 1]++] = considerd_char;
        }
      }
      if ((size & 63ULL) && ((omp_rank + 1) == omp_size)) {
        for (uint64_t i = size - (size & 63ULL); i < size; ++i) {
          const AlphabetType considerd_char = (text[i] >> cur_bit_shift);
          sorted_text[borders_aligned[considerd_char >> 1]++] = considerd_char;
        }
      }

      #pragma omp barrier

      //we need to write to all bit vectors of the current level
      omp_write_bits_level(level, bv, [&](uint64_t pos) {
        return (sorted_text[pos] & 1ULL) != 0;
      });
    }
  }
}

template <bool requires_initialization = true>
class empty_vectors {
public:
    template<typename T>
    empty_vectors(uint64_t, T) {}

    auto operator[](const uint64_t index) {
        DCHECK(false);
    }

    auto operator[](const uint64_t index) const {
        DCHECK(false);
    }
};

template <typename sym_t, typename idx_t>
inline void wt_pps_nodebased(wt_bits_t& bits, const std::vector<sym_t>& text, const size_t h) {
    using ctx_t = ctx_generic<true,
                            ctx_options::borders::sharded_single_level,
                            ctx_options::hist::sharded_single_level,
                            ctx_options::live_computed_rho,
                            ctx_options::bv_initialized,
                            empty_vectors>;
    
    const uint64_t shards = omp_get_max_threads();
    ctx_t ctx(text.size(), h, h, shards);
    pps(text.data(), text.size(), h, ctx, bits);
}

// prefix sorting
template <typename sym_t, typename idx_t>
inline void
wt_pps_nodebased(const WaveletTreeBase& wt, wt_bits_t& bits, const std::vector<sym_t>& text) {
    wt_pps_nodebased<sym_t, idx_t>(bits, text, wt.height());
}
