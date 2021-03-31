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
    const std::string s = "abracadabra";
    std::vector<uint8_t> vec(s.begin(), s.end());
    std::cout << "input:";
    for (auto i = vec.begin(); i != vec.end(); ++i)
        std::cout << static_cast<int>(*i) << ' ';
    std::cout << '\n';
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
void do_compute(const std::vector<uint8_t>& vec) {
    const uint64_t levels = no_reduction_alphabet(vec);
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
    mpi_dd::template start<uint8_t>(ctx, "test_input.txt", SIZE_MAX /* prefix */, 0 /* rdbufsize */,
                                    false /* effective input */, "" /* Output */);
}

int main(int argc, char* argv[]) {
    //computeDist(argc, argv);
    std::vector<uint8_t> input;
    if (argc < 2) {
        input = get_small_input();
    } else {
        input = get_file_input(argv[1]);
    }
    do_compute<wt_pc_pwm>(input);
    return EXIT_SUCCESS;
}
