#include <iostream>
#include <string>
#include <vector>
#include <bitset>
#include <mpi.h>
#include "construction/pc.hpp"
#include "util/alphabet_util.hpp"
#include "util/decode.hpp"
#include "wx_pc.hpp"
#include "wx_ppc.hpp"

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

template<typename Algorithm>
void do_compute(const std::vector<uint8_t>& vec)
{
    const uint64_t levels = no_reduction_alphabet(vec);
    Algorithm tree;
    const auto output = tree.compute(vec.data(), vec.size(), levels);

    const auto &bvs = output.bvs();
    std::cout << "levels: " << bvs.levels() << '\n';
    for (size_t i = 0; i < bvs.levels(); i++)
    {
        std::cout << "level " << i << " level_bit_size: " << bvs.level_bit_size(i) << '\n';
        const auto level = bvs[i];
        const auto v = level[0];
        std::cout << std::bitset<64>(v) << '\n';
    }

    const auto decoded = decode_wt(bvs, vec.size());
    std::cout << "decoded: " << decoded << '\n';
}

int main()
{
    const auto input = get_small_input();
    do_compute<wt_ppc>(input);
    return EXIT_SUCCESS;
}
