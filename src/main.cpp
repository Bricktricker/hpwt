#include <iostream>
#include <string>
#include <vector>
#include <bitset>
#include <mpi.h>
#include "construction/pc.hpp"
#include "util/alphabet_util.hpp"
#include "util/decode.hpp"
#include "wx_pc.hpp"

int main()
{
    const std::string s = "abracadabra";
    auto vec = std::vector<uint8_t>(s.begin(), s.end());
    std::cout << "input:";
    for (auto i = vec.begin(); i != vec.end(); ++i)
        std::cout << static_cast<int>(*i) << ' ';
    std::cout << '\n';
    const uint64_t levels = no_reduction_alphabet(vec);
    wx_pc<uint8_t, true> tree;
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

    return EXIT_SUCCESS;
}
