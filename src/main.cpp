#include "construction/pc.hpp"
#include "util/alphabet_util.hpp"
#include "util/decode.hpp"
#include "wx_pc.hpp"
#include "wx_ppc.hpp"
#include <bitset>
#include <fstream>
#include <iostream>
#include <iterator>
#include <mpi.h>
#include <string>
#include <vector>

using wt_pc = wx_pc<uint8_t, true>;
using wt_ppc = wx_ppc<uint8_t, true>;

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

int main(int argc, char* argv[]) {
    std::vector<uint8_t> input;
    if (argc < 2) {
        input = get_small_input();
    } else {
        input = get_file_input(argv[1]);
    }
    do_compute<wt_ppc>(input);
    return EXIT_SUCCESS;
}
