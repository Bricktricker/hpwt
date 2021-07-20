#include <array>
#include <bitset>
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

template <typename A, size_t N>
void pack_new(A* _dst, const A* src, size_t start, size_t end) {
    span<A> dst(_dst, required_bufsize_temp<N>(end));

    const size_t start_block = start / N; // the first block to write into (inclusive)
    const size_t end_block = end / N;     // the last block to write into (inclusive)
    const size_t block_offs = start % N;

    // start is at start of block, we can easily copy the bits over
    if (block_offs == 0) {

        for (size_t block = start_block; block < end_block; block++) {
            const size_t src_block = block - start_block;
            dst[block] = src[src_block];
        }

        const auto bits_left = end % N;
        if (bits_left != 0) {
            const auto bits_left_mask = ((1ULL << bits_left) - 1) << (N - bits_left);
            dst[end_block] |= (src[end_block - start_block] & bits_left_mask) >> block_offs;
        }

    } else {
        const size_t inv_block_offs = N - block_offs;
        const auto mask = (1ULL << block_offs) - 1; // sets lowest block_offs bits to 1

        // extra handling wen we need to copy less than N bits
        if (end - start < N - block_offs) {
            const size_t bits_left = (end - start) % N;
            const auto bits_left_mask = ((1ULL << bits_left) - 1) << (N - bits_left);
            dst[start_block] |= static_cast<A>((src[0] & bits_left_mask) >> block_offs);
            return;
        } else {
            dst[start_block] |= static_cast<A>(src[0] >> block_offs);
        }

        for (size_t block = start_block + 1; block < end_block; block++) {
            size_t src_block = (block - start_block) - 1;
            dst[block] |= (src[src_block] & mask) << inv_block_offs;

            src_block++;
            dst[block] |= src[src_block] >> block_offs;
        }

        const size_t bits_left = end % N;
        if (bits_left != 0) {
            const size_t src_block = (end_block - start_block) - 1;
            const auto bits_left_mask = ((1ULL << bits_left) - 1) << (N - bits_left);
            dst[end_block] |= (src[src_block] & bits_left_mask);
        }
    }
}

int main() {
    const size_t start = 5;
    const size_t end = 21;
    std::array<uint8_t, 3> src({0xFF, 0xFF, 0xFF}); // 0b10010100, 0b10101001, 0b11100101
    const size_t buff_size = required_bufsize_temp<8>(end);
    uint8_t* dst = new uint8_t[buff_size];
    std::memset(dst, 0, buff_size);
    pack_new<uint8_t, 8>(dst, src.data(), start, end);
    std::cout << "dst: ";
    print_binary<uint8_t, 8>(dst, buff_size);
    return 0;
}