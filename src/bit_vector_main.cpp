#include <array>
#include <bitset>
#include <distwt/mpi/uint64_pack_bv64.hpp>
#include <iostream>
#include <pwm/arrays/span.hpp>
#include <pwm/util/debug_assert.hpp>
#include <src/bit_vector.hpp>
#include <src/omp_write_bits.hpp>
#include <tlx/math/div_ceil.hpp>

template <size_t N>
size_t required_bufsize_temp(const size_t num_items) {
    return tlx::div_ceil(num_items, N);
}

template <typename A, size_t N>
void print_binary(const A* ptr, const size_t length) {
    for (size_t i = 0; i < length; i++) {
        std::bitset<N> x(ptr[i]);
        std::cout << x << ' ';
    }
    std::cout << '\n';
}

void old_copy(bit_vector& vec, const uint64_t start, const uint64_t end, const uint64_t* msg) {
    omp_write_bits_vec(start, end, vec, [&](uint64_t const idx) {
        const size_t local_idx = idx - start;
        const uint64_t word = msg[2 + (local_idx / 64)];
        const size_t bit_idx = 64 - (local_idx % 64) - 1;
        uint64_t const bit = ((word >> bit_idx) & 1ULL);
        return bit != 0;
    });
}

void test_1() {
    bit_vector vec_1;
    bit_vector vec_2;
    vec_1.resize(88);
    vec_2.resize(88);

    const uint64_t b1 = 11894728207041305296ULL;
    const uint64_t b2 = 9231347388919854438ULL;

    omp_copy_bits(vec_1, &b1, 0, 62);
    omp_copy_bits(vec_1, &b2, 62, 88);

    std::array<uint64_t, 3> arr_1{0, 0, b1};
    std::array<uint64_t, 3> arr_2{0, 0, b2};
    old_copy(vec_2, 0, 62, arr_1.data());
    old_copy(vec_2, 62, 88, arr_2.data());
    
    if(vec_1 != vec_2) {
        std::cout << vec_1 << '\n';
        std::cout << vec_2 << '\n';
        throw  std::runtime_error("test_1 failed");
    }
}

void test_2() {
    bit_vector vec_1;
    bit_vector vec_2;
    vec_1.resize(119);
    vec_2.resize(119);

    const uint64_t b1 = 16815872969109667840ULL;
    const uint64_t b2 = 15036473865976938496ULL;
    const std::array<uint64_t, 2> b3{18113243328543154591ULL, 5044031582654955520ULL};

    omp_copy_bits(vec_1, &b1, 0, 43);
    omp_copy_bits(vec_1, &b2, 114, 119);
    omp_copy_bits(vec_1, b3.data(), 43, 114);

    std::array<uint64_t, 3> arr_1{0, 0, b1};
    std::array<uint64_t, 3> arr_2{0, 0, b2};
    std::array<uint64_t, 4> arr_3{0, 0, b3[0], b3[1]};
    old_copy(vec_2, 0, 43, arr_1.data());
    old_copy(vec_2, 114, 119, arr_2.data());
    old_copy(vec_2, 43, 114, arr_3.data());

    if(vec_1 != vec_2) {
        std::cout << vec_1 << '\n';
        std::cout << vec_2 << '\n';
        throw std::runtime_error("test_1 failed");
    }

}

int main() {
    test_1();
    test_2();
    return 0;
}