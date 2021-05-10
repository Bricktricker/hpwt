#pragma once

#include <distwt/common/effective_alphabet.hpp>
#include <distwt/common/wt.hpp>

#include <algorithm>
#include <cassert>
#include <omp.h>
#include <tlx/math/integer_log2.hpp>

// one bit vector per node
using wt_bits_t = std::vector<std::vector<bool>>;

// prefix counting for wavelet subtree
inline void omp_write_bits_vec(uint64_t start,
                               uint64_t size,
                               bv_t& level_bv,
                               std::function<bool(uint64_t)> body) {
    const auto omp_rank = omp_get_thread_num();
    const auto omp_size = omp_get_num_threads();

#pragma omp for
    for (int64_t scur_pos = start; scur_pos <= (int64_t(size) - 64); scur_pos += 64) {
        DCHECK(scur_pos >= 0);
        for (size_t i = 0; i < 64; i++) {
            level_bv[scur_pos + i] = body(scur_pos + i);
        }
    }

    uint64_t const remainder = size & 63ULL;
    if (remainder && ((omp_rank + 1) == omp_size)) {
        const auto scur_pos = size - remainder;
        for (size_t i = 0; i < remainder; i++) {
            level_bv[scur_pos + i] = body(scur_pos + i);
        }
    }
}

template <typename sym_t, typename idx_t>
inline void wt_pc_combined(wt_bits_t& bits, // bits ist ein vector<Bitvector> und enthält für jede
                                            // knoten im baum einen Bitvektor
                           const std::vector<sym_t>& text,
                           const size_t root_node_id, // 1-based!!
                           const size_t h) {

    assert(root_node_id > 0);
    const size_t root_level = tlx::integer_log2_floor(root_node_id);
    const size_t root_rank = (root_node_id - (1ULL << root_level));
    const size_t glob_h = root_level + h;

    const size_t n = text.size();
    const size_t sigma = 1ULL << h; // we need the next power of two!

    assert(h >= 1);

    // compute initial histogram
    std::vector<idx_t> hist(sigma, 0);
    {
        auto& root = bits[root_node_id - 1];
        root.resize(n);

        // Allocate a histogram buffer per thread
        helper_array sharded_hists(omp_get_max_threads(), sigma);

// Write the first level, and fill the histograms in parallel
#pragma omp parallel
        {
            const auto shard = omp_get_thread_num();
            auto&& hist = sharded_hists[shard];
            omp_write_bits_vec(0, n, root, [&](uint64_t const i) {
                auto const c = text[i];
                hist[c]++;
                uint64_t const bit = ((c >> (h - 1)) & 1ULL);
                return bit != 0;
            });
        }

        // Accumulate the histograms
#pragma omp parallel for
        for (uint64_t j = 0; j < sigma; ++j) {
            for (uint64_t shard = 0; shard < sharded_hists.levels(); ++shard) {
                hist[j] += sharded_hists[shard][j];
            }
        }
    }

    // compute histogram and root node
    for (size_t level = h - 1; level > 0; --level) {
        const size_t num_level_nodes = (1ULL << level);
        const size_t glob_offs = ((1ULL << level) * root_node_id) - 1;
        for (size_t v = 0; v < num_level_nodes; v++) {
            const size_t size = hist[2 * v] + hist[2 * v + 1];
            const size_t node = glob_offs + v;

            hist[v] = size;
            bits[node].resize(size);
        }
    }

    // allocate counters
    std::vector<std::vector<idx_t>> sharded_counter(omp_get_max_threads());
    for (auto& vec : sharded_counter) {
        vec.resize(sigma / 2);
    }

#pragma omp parallel for schedule(nonmonotonic : dynamic, 1)
    for (size_t level = h - 1; level > 0; --level) {
        auto& count = sharded_counter[omp_get_thread_num()];
        std::fill(count.begin(), count.end(), 0); // reset counters

        const size_t glob_level = root_level + level;
        const size_t glob_offs = ((1ULL << level) * root_node_id) - 1;

        // compute level bit vectors
        const size_t rsh = glob_h - 1 - (glob_level - 1);
        const size_t test = 1ULL << (glob_h - 1 - glob_level);

        for (size_t i = 0; i < n; i++) {
            const size_t c = text[i];
            const size_t glob_v = (c >> rsh);
            const size_t v = glob_v - root_rank * (1ULL << level);

            // assert(glob_v >= root_rank * (1ULL << level));
            // assert(v < num_level_nodes);
            const size_t node = glob_offs + v;

            const size_t pos = count[v];
            ++count[v];
            const bool b = c & test;

            // assert(pos < hist[v]);
            bits[node][pos] = b;
        }
    }
}

// prefix counting
template <typename sym_t, typename idx_t>
inline void
wt_pc_combined(const WaveletTreeBase& wt, wt_bits_t& bits, const std::vector<sym_t>& text) {

    wt_pc_combined<sym_t, idx_t>(bits, text, 1, wt.height());
}
