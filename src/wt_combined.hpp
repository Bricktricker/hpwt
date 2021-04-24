#pragma once

#include <distwt/common/effective_alphabet.hpp>
#include <distwt/common/wt.hpp>

#include <cassert>
#include <omp.h>
#include <tlx/math/integer_log2.hpp>

// one bit vector per node
using wt_bits_t = std::vector<std::vector<bool>>;

using ctx_t = ctx_generic<true,
                          ctx_options::borders::sharded_single_level,
                          ctx_options::hist::all_level,
                          ctx_options::pre_computed_rho,
                          ctx_options::bv_initialized,
                          bit_vectors>;

// prefix counting for wavelet subtree
template <typename sym_t, typename idx_t>
inline void wt_pc_combined(wt_bits_t& bits, // bits ist ein vector<Bitvector> und enthält für jede
                                            // knoten im baum einen Bitvektor
                           const std::vector<sym_t>& text,
                           const size_t root_node_id, // 1-based!!
                           const size_t h) {

    assert(root_node_id > 0);
    const size_t root_level = tlx::integer_log2_floor(root_node_id); // 0
    const size_t root_rank = (root_node_id - (1ULL << root_level));  // 0
    const size_t glob_h = root_level + h;                            // 3 = 0 + 3

    const size_t n = text.size();   // 20
    const size_t sigma = 1ULL << h; // 8, we need the next power of two!

    const auto rho = rho_dispatch<true>::create(h);
    ctx_t ctx(n, h, rho, h);

    assert(h >= 1);

    std::cout << "Using " << omp_get_max_threads() << " omp threads\n";

    // compute initial histogram
    {
        const uint64_t alphabet_size = sigma;
        auto&& bv = ctx.bv();

        auto& root = bits[root_node_id - 1];
        root.resize(n);

        // Allocate a histogram buffer per thread
        helper_array sharded_hists(omp_get_max_threads(), alphabet_size);

// Write the first level, and fill the histograms in parallel
#pragma omp parallel
        {
            const auto shard = omp_get_thread_num();
            auto&& hist = sharded_hists[shard];

            // schreibt aktuell die bits des ersten levels in bv[0]
            omp_write_bits_wordwise(0, n, bv[0], [&](uint64_t const i) {
                auto const c = text[i];
                hist[c]++;
                uint64_t const bit = ((c >> (h - 1)) & 1ULL);
                return bit;
            });
        }

        // Accumulate the histograms
        auto&& hist = ctx.hist_at_level(h);
#pragma omp parallel for
        for (uint64_t j = 0; j < alphabet_size; ++j) {
            for (uint64_t shard = 0; shard < sharded_hists.levels(); ++shard) {
                hist[j] += sharded_hists[shard][j];
            }
        }

        // copy bits from bv[0] to root
        for (size_t i = 0; i < n; i++) {
            root[i] = bit_at(bv[0], i);
        }
    }

    auto&& leaf_hist = ctx.hist_at_level(h);

// Compute the histogram for each level of the wavelet structure
#pragma omp parallel for schedule(nonmonotonic : dynamic, 1)
    for (uint64_t level = 1; level < h; ++level) {
        auto&& hist = ctx.hist_at_level(level);
        const uint64_t blocks = (1 << level);
        const uint64_t required_characters = (1 << (h - level));
        for (uint64_t i = 0; i < blocks; ++i) {
            for (uint64_t j = 0; j < required_characters; ++j) {
                hist[i] += leaf_hist[(i * required_characters) + j];
            }
        }
    }

    // compute histogram and root node
    auto&& hist = ctx.hist_at_level(h);

    // resize vectors in bits
    for (size_t level = h - 1; level > 0; --level) {
        const size_t num_level_nodes = (1ULL << level);
        const size_t glob_offs =
            ((1ULL << level) * root_node_id) - 1; // anzahl nodes in den vorherigen Ebenen(?)
        for (size_t v = 0; v < num_level_nodes; v++) {
            const size_t size = hist[2 * v] + hist[2 * v + 1];
            const size_t node = glob_offs + v;

            hist[v] = size;
            bits[node].resize(size);
        }
    }

    // allocate counters
    std::vector<std::vector<idx_t>> sharded_counter(omp_get_max_threads());
    for(auto& vec : sharded_counter) {
        vec.resize(sigma / 2);
    }

#pragma omp parallel for schedule(nonmonotonic : dynamic, 1)
    for (size_t level = h - 1; level > 0; --level) {        
        auto& count = sharded_counter[omp_get_thread_num()];
        std::memset(count.data(), 0, count.size()); // reset counters

        const size_t glob_level = root_level + level;
        const size_t glob_offs =
            ((1ULL << level) * root_node_id) - 1; // anzahl nodes in den vorherigen Ebenen(?)

        // compute level bit vectors
        const size_t rsh =
            glob_h - 1 - (glob_level - 1); // cur_bit_shift aus building_block#write_symbol_bit ?
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

    std::cout << '\n';
}

// prefix counting
template <typename sym_t, typename idx_t>
inline void
wt_pc_combined(const WaveletTreeBase& wt, wt_bits_t& bits, const std::vector<sym_t>& text) {

    wt_pc_combined<sym_t, idx_t>(bits, text, 1, wt.height());
}
