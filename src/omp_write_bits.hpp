#pragma once
#include <vector>
#include <cstdint>
#include <omp.h>

// one bit vector per node
using wt_bits_t = std::vector<std::vector<bool>>;

template <typename loop_body_t>
inline void omp_write_bits_vec(uint64_t start, uint64_t size, bv_t& level_bv, loop_body_t body) {
#pragma omp for
    for (int64_t scur_pos = start; scur_pos <= (int64_t(size) - 64); scur_pos += 64) {
        DCHECK(scur_pos >= 0);
        for (size_t i = 0; i < 64; i++) {
            level_bv[scur_pos + i] = body(scur_pos + i);
        }
    }

    const auto omp_rank = omp_get_thread_num();
    const auto omp_size = omp_get_num_threads();

    uint64_t const remainder = size & 63ULL;
    if (remainder && ((omp_rank + 1) == omp_size)) {
        const auto scur_pos = size - remainder;
        for (size_t i = 0; i < remainder; i++) {
            level_bv[scur_pos + i] = body(scur_pos + i);
        }
    }
}

template <typename loop_body_t>
inline void omp_write_bits_level(const uint64_t level, wt_bits_t& bits, loop_body_t body) {
    const size_t num_nodes_level = 1ULL << level;
    const size_t nodes_offset = num_nodes_level - 1;

#pragma omp for
    for(size_t node = 0; node < num_nodes_level; node++) {
      const size_t glob_node = nodes_offset + node;
      const size_t num_bits = bits[glob_node].size();
      
      //count number of bits in previous nodes
      const size_t bits_offset = std::accumulate(
        std::next(bits.begin(), nodes_offset),
        std::next(bits.begin(), glob_node),
        0,
        [](const size_t acc, const auto& vec) {
          return acc + vec.size();
        }
      );
      
      for(size_t i = 0; i < num_bits; i++) {
        bits[glob_node][i] = body(bits_offset + i);
      }
    }
}
