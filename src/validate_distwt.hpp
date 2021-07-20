#include "bit_rank.hpp"
#include <distwt/common/binary_io.hpp>
#include <distwt/common/bv64.hpp>
#include <distwt/common/effective_alphabet.hpp>
#include <distwt/common/util.hpp>
#include <distwt/mpi/wt.hpp>
#include <iomanip>
#include <string>
#include <tlx/math/div_ceil.hpp>

template <typename sym_t>
static void
validate_distwt(const std::string& input, const std::string& output, const size_t comm_size, const size_t prefix) {
    const size_t input_size = std::min(util::file_size(input), prefix) / sizeof(sym_t);
    const auto size_per_worker = tlx::div_ceil(input_size, comm_size);

    Histogram<sym_t> hist(output + "." + WaveletTreeBase::histogram_extension());
    EffectiveAlphabetBase<sym_t> ea(hist);

    const size_t tree_height =
        tlx::integer_log2_ceil(hist.size() - 1); // WaveletTreeBase::wt_height

    // read wavelet tree
    std::vector<std::vector<bool>> wt;
    for (size_t level = 0; level < tree_height; level++) {
        wt.emplace_back();
        size_t bits_left = input_size;
        for (size_t rank = 0; rank < comm_size; rank++) {
            // construct local filename
            std::string filename;
            {
                std::ostringstream ss;
                ss << output << std::setw(4) << std::setfill('0') << rank << '.'
                   << WaveletTreeBase::level_extension(level);
                filename = ss.str();
            }

            binary::FileReader reader(filename);
            bv64_t bitbuf(reader.read<uint64_t>());
            size_t x = 0;
            for (size_t i = 0; i < std::min(size_per_worker, bits_left); i++) {
                if (x >= 64ULL) {
                    bitbuf = bv64_t(reader.read<uint64_t>());
                    x = 0;
                }
                wt.back().push_back(bitbuf[63ULL - (x++)]);
            }

            bits_left -= std::min(size_per_worker, bits_left);
        }
    }

    std::vector<bit_rank> level_ranks;
    for (size_t level = 0; level < tree_height; level++) {
        level_ranks.emplace_back(wt[level]);
    }

    // reconstruct input file and compare with input file
    binary::FileReader file_reader(input);
    for (size_t i = 0; i < input_size; i++) {
        sym_t value = 0;
        size_t idx = i;
        size_t level_begin = 0;
        size_t level_end = input_size;
        for (size_t level = 0; level < tree_height; level++) {
            const auto& level_bits = wt[level];
            const size_t begin_zeros =
                level_begin > 0 ? level_ranks[level].rank0(level_begin - 1) : 0;
            const size_t num_level_zeros = level_ranks[level].rank0(level_end - 1) - begin_zeros;

            const auto bit = level_bits[idx];
            value <<= 1;
            value |= bit;
            if (!bit) {
                // go left
                // count number of 0's from level_begin to idx
                idx = idx > 0 ? level_ranks[level].rank0(idx - 1) - begin_zeros : 0;

                level_end = level_begin + num_level_zeros;
            } else {
                // go right
                // count number of 1's from level_begin to idx
                const size_t begin_ones =
                    level_begin > 0 ? level_ranks[level].rank1(level_begin - 1) : 0;
                idx = idx > 0 ? level_ranks[level].rank1(idx - 1) - begin_ones : 0;

                level_begin = level_begin + num_level_zeros;
            }
            idx += level_begin;
        }

        // read char from file
        const sym_t file_value = ea.map(file_reader.read<sym_t>());
        if (file_value != value) {
            // decoded and file value mismatch
            std::cerr << "Error while decoding wavelet tree on position " << i << ", expected "
                      << static_cast<size_t>(file_value) << ", but got "
                      << static_cast<size_t>(value) << '\n';
        }
    }
}