#pragma once
#include <cstdint>
#include <distwt/mpi/wt.hpp>
#include <omp.h>
#include <tlx/math/div_ceil.hpp>
#include <vector>

// one bit vector per node
using wt_bits_t = WaveletTree::bits_t;

// start = offset in bits into dst, end = last bit + 1 position to write into dst, src reading
// allways starts at 0
inline void omp_copy_bits(bv_t& dst, const uint64_t* src, size_t start, const size_t end) {
    DCHECK_LE(end, dst.size());
    const auto omp_rank = omp_get_thread_num();
    const auto omp_size = omp_get_num_threads();

    constexpr size_t N = 64ULL;

    const size_t start_block = start / N; // the first block to write into (inclusive)
    const size_t end_block = end / N;     // the last block to write into (inclusive)
    const size_t block_offs = start % N;

    // start is at start of block, we can easily copy the bits over
    if (block_offs == 0) {

#pragma omp for
        for (size_t block = start_block; block < end_block; block++) {
            const size_t src_block = block - start_block;
            dst.data()[block] = src[src_block];
        }

        const auto bits_left = end % N;
        if (bits_left != 0 && omp_rank == 0) {
            const auto bits_left_mask = ((1ULL << bits_left) - 1) << (N - bits_left);
            dst.data()[end_block] |= (src[end_block - start_block] & bits_left_mask) >> block_offs;
        }

    } else {
        const size_t inv_block_offs = N - block_offs;
        const auto mask = (1ULL << block_offs) - 1; // sets lowest block_offs bits to 1

        // extra handling when we need to copy less than N bits
        if (end - start < N - block_offs) {
            if (omp_rank == 0) {
                const size_t bits_left = (end - start) % N;
                const auto bits_left_mask = ((1ULL << bits_left) - 1) << (N - bits_left);
                dst.data()[start_block] |= (src[0] & bits_left_mask) >> block_offs;
            }
            return;
        } else if (omp_rank == 0) {
            // copy the higest inv_block_offs bits from src[0] into the lowest bits from dst
            dst.data()[start_block] |= src[0] >> block_offs;
        }

#pragma omp for
        for (size_t block = start_block + 1; block < end_block; block++) {
            size_t src_block = (block - start_block) - 1;
            dst.data()[block] |= (src[src_block] & mask) << inv_block_offs;

            src_block++;
            dst.data()[block] |= src[src_block] >> block_offs;
        }

        const size_t bits_left = end % N;
        if (bits_left != 0 && ((omp_rank + 1) == omp_size)) {
            if (bits_left + inv_block_offs > N) {
                size_t src_block = (end_block - start_block) - 1;

                // copy the lowest block_offs bits from src_block into the highest block_offs bits of dst
                dst.data()[end_block] |= (src[src_block] & mask) << inv_block_offs;

                src_block++;

                // mask that has the highest (bits_left - block_offs) bits set
                const auto bits_left_mask = ((1ULL << (bits_left - block_offs)) - 1) << (N - (bits_left - block_offs));

                dst.data()[end_block] |= (src[src_block] & bits_left_mask) >> block_offs;
            } else {
                const size_t src_block = (end_block - start_block) - 1;
                const auto bits_left_mask = ((1ULL << bits_left) - 1)
                                            << (N - bits_left - inv_block_offs);
                dst.data()[end_block] |= (src[src_block] & bits_left_mask) << inv_block_offs;
            }
        }
    }
}

template <typename loop_body_t>
inline void omp_write_bits_vec(uint64_t start, uint64_t end, bv_t& level_bv, loop_body_t body) {
    const auto omp_rank = omp_get_thread_num();
    const auto omp_size = omp_get_num_threads();

    constexpr size_t SPACING = 512ULL; //Bits

    uint64_t const start_filler = start & (SPACING-1);
    if (start_filler) {
        const size_t end_fill = std::min(uint64_t(SPACING) - start_filler, end - start);
        if ((omp_rank + 1) == omp_size) {
            for (size_t i = 0; i < end_fill; i++) {
                level_bv.set(start + i, body(start + i));
            }
        }
        start += end_fill;
    }

#pragma omp for
    for (int64_t scur_pos = start; scur_pos <= (int64_t(end) - int64_t(SPACING)); scur_pos += int64_t(SPACING)) {
        DCHECK(scur_pos >= 0);
        for(size_t block = 0; block < SPACING; block += 64) {
            // fills the 64 bit block with bits, than write that into the bit vector
            uint64_t val = 0;
            for (size_t i = 0; i < 64; i++) {
                const bool bit = body(scur_pos + block + i);
                val |= static_cast<uint64_t>(bit) << (64 - i - 1);
            }
            level_bv.data()[(scur_pos + block) / 64] = val;
        }
    }

    uint64_t const remainder = (end - start) & (SPACING-1);
    if (remainder && ((omp_rank + 1) == omp_size)) {
        const auto scur_pos = end - remainder;
        for (size_t i = 0; i < remainder; i++) {
            DCHECK((scur_pos + i) >= start);
            DCHECK((scur_pos + i) < end);
            level_bv.set(scur_pos + i, body(scur_pos + i));
        }
    }
}

template <typename loop_body_t>
inline void omp_write_bits_level(const uint64_t level, wt_bits_t& bits, loop_body_t body) {
    const size_t num_nodes_level = 1ULL << level;
    const size_t nodes_offset = num_nodes_level - 1;

    size_t bits_offset = 0;
    for (size_t node = 0; node < num_nodes_level; node++) {
        const size_t glob_node = nodes_offset + node;
        const size_t num_bits = bits[glob_node].size();

        omp_write_bits_vec(0, num_bits, bits[glob_node], [&](const uint64_t pos) {
            return body(bits_offset + pos);
        });
        bits_offset += num_bits;
    }
}
