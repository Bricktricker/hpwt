#include <distwt/common/binary_io.hpp>
#include <distwt/common/util.hpp>
#include <distwt/mpi/wt.hpp>
#include <iomanip>
#include <string>
#include <tlx/math/div_ceil.hpp>

template <typename sym_t>
static bool validate_distwt(const std::string& input,
                            const std::string& output,
                            const size_t comm_size,
                            const size_t tree_height) {
    const size_t input_size = util::file_size(input) / sizeof(sym_t);
    const auto size_per_worker = tlx::div_ceil(input_size, comm_size);

    // read wavelet tree
    WaveletTree::bits_t wt;
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

    if (wt[0].size() <= 64) {
        std::cout << "loaded tree:\n";
        for (const auto level : wt) {
            std::cout << level << '\n';
        }
    }

    Histogram<sym_t> hist(output + "." + WaveletTreeBase::histogram_extension());
    EffectiveAlphabetBase<sym_t> ea(hist);

    // reconstruct input file and compare with input file
    binary::FileReader file_reader(input);
    for (size_t i = 0; i < input_size; i++) {
        sym_t value = 0;
        size_t idx = i;
        size_t level_begin = 0;
        size_t level_end = input_size;
        for (size_t level = 0; level < tree_height; level++) {
            const auto& level_bits = wt[level];
            const size_t num_level_zeros = std::accumulate(
                std::next(level_bits.begin(), level_begin),
                std::next(level_bits.begin(), level_end), 0, [](const size_t acc, const bool v) {
                    return v ? acc : acc + 1;
                }); // increment acc on false to count 0's

            const auto bit = level_bits[idx];
            value <<= 1;
            value |= bit;
            if (!bit) {
                // go left
                // count number of 0's from level_begin to idx
                idx = std::accumulate(std::next(level_bits.begin(), level_begin),
                                      std::next(level_bits.begin(), idx), 0,
                                      [](const size_t acc, const bool v) {
                                          return v ? acc : acc + 1;
                                      }); // increment acc on false to count 0's

                level_end = level_begin + num_level_zeros;
            } else {
                // go right
                // count number of 1's from level_begin to idx
                idx = std::accumulate(std::next(level_bits.begin(), level_begin),
                                      std::next(level_bits.begin(), idx), 0);

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

    return true;
}