#pragma once

#include <distwt/common/effective_alphabet.hpp>
#include <distwt/common/wt.hpp>
#include <pwm/arrays/flat_two_dim_array.hpp>
#include <pwm/arrays/helper_array.hpp>
#include <pwm/util/debug.hpp>

#include <algorithm>
#include <cassert>
#include <omp.h>
#include <tlx/math/integer_log2.hpp>
#include <src/omp_write_bits.hpp>

class wt_ppc_nodebased {
public:

// prefix counting for wavelet subtree
// combination of wt_pc and ppc
template <typename sym_t, typename idx_t>
static void start(wt_bits_t& bits, const std::vector<sym_t>& text, const size_t h) {

    const size_t n = text.size();
    const size_t sigma = 1ULL << h; // we need the next power of two!

    assert(h >= 1);

    // compute initial histogram
    std::vector<idx_t> hist(sigma, 0);
    {
        auto& root = bits[0];
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
        const size_t glob_offs = (1ULL << level) - 1;
        for (size_t v = 0; v < num_level_nodes; v++) {
            const size_t size = hist[2 * v] + hist[2 * v + 1];
            const size_t node = glob_offs + v;

            hist[v] = size;
            bits[node].resize(size);
        }
    }

#pragma omp parallel num_threads(std::min(h - 1, static_cast<size_t>(omp_get_max_threads())))
    {
        std::vector<idx_t> count;
#pragma omp for schedule(nonmonotonic : dynamic, 1)
        for (size_t level = h - 1; level > 0; --level) {
            if (count.empty()) {
                count.resize(sigma / 2, 0);
            }else{
                std::fill(count.begin(), count.end(), 0); // reset counters
            }

            const size_t glob_offs = (1ULL << level) - 1;

            // compute level bit vectors
            const size_t rsh = h - 1 - (level - 1);
            const size_t test = 1ULL << (h - 1 - level);

            for (size_t i = 0; i < n; i++) {
                const size_t c = text[i];
                const size_t v = (c >> rsh);

                const size_t node = glob_offs + v;

                const size_t pos = count[v];
                ++count[v];
                const bool b = c & test;

                // assert(pos < hist[v]);
                bits[node][pos] = b;
            }
        }
    }
}

// prefix counting
template <typename sym_t, typename idx_t>
static void
start(const WaveletTreeBase& wt, wt_bits_t& bits, const std::vector<sym_t>& text) {
    start<sym_t, idx_t>(bits, text, wt.height());
}

static std::string name() {
  return "ppc";
}

};
