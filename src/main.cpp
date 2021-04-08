#include <bitset>
#include <distwt/apps/mpi_dd.hpp>
#include <distwt/mpi/context.hpp>
#include <distwt/mpi/uint_types.hpp>
#include <fstream>
#include <iostream>
#include <iterator>
#include <mpi.h>
#include <pwm/util/alphabet_util.hpp>
#include <pwm/util/decode.hpp>
#include <pwm/wx_pc.hpp>
#include <pwm/wx_ppc.hpp>
#include <string>
#include <vector>

using wt_pc_pwm = wx_pc<uint8_t, true>;
using wt_ppc_pwm = wx_ppc<uint8_t, true>;

std::vector<uint8_t> get_small_input() {
    const std::string s = "abcdabcdefefefghghab";
    std::vector<uint8_t> vec(s.begin(), s.end());
    return vec;
}

std::vector<uint8_t> get_file_input(const std::string& filePath) {
    std::ifstream is(filePath);
    if (!is) {
        std::cerr << "Could not open file " << filePath << '\n';
        throw std::runtime_error("Could not open file");
    }
    const std::vector<uint8_t> data((std::istreambuf_iterator<char>(is)),
                                    std::istreambuf_iterator<char>());
    return data;
}

template <typename Algorithm>
void do_compute(std::vector<uint8_t>& vec) {
#define REDUCE_VEV
#ifdef REDUCE_VEV
    const auto max_char = reduce_textbook(vec);
    const uint64_t levels = levels_for_max_char(max_char);
#else
    const uint64_t levels = no_reduction_alphabet(vec);
#endif
    std::cout << "input:";
    for (auto i = vec.begin(); i != vec.end(); ++i)
        std::cout << static_cast<int>(*i) << ' ';
    std::cout << '\n';
    Algorithm tree;
    const auto output = tree.compute(vec.data(), vec.size(), levels);

    const auto& bvs = output.bvs();
    std::cout << "Algorithm: " << typeid(tree).name() << '\n';
    std::cout << "levels: " << bvs.levels() << '\n';
    for (size_t i = 0; i < bvs.levels(); i++) {
        std::cout << "level " << i << " level_bit_size: " << bvs.level_bit_size(i) << '\n';
        if (bvs.level_bit_size(i) <= 64) {
            const auto level = bvs[i];
            const auto v = level[0];
            std::cout << std::bitset<64>(v) << '\n';
        }
    }

    if (vec.size() <= 64) {
        const auto decoded = decode_wt(bvs, vec.size());
        std::cout << "decoded: " << decoded << '\n';
    }
}

void computeDist(int argc, char* argv[]) {
    MPIContext ctx(&argc, &argv);
    mpi_dd::template start<uint8_t>(ctx, "small_input_file.bin", SIZE_MAX /* prefix */, 0 /* rdbufsize */,
                                    false /* effective input */, "output/out" /* Output */);
}

int main(int argc, char* argv[]) {
#define USE_MPI
#ifdef USE_MPI
    computeDist(argc, argv);
#else
    std::vector<uint8_t> input;
    if (argc < 2) {
        input = get_small_input();
    } else {
        input = get_file_input(argv[1]);
    }
    do_compute<wt_pc_pwm>(input);
#endif
    return EXIT_SUCCESS;
}
