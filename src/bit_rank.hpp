#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <vector>

/// \brief A space efficient data structure for answering rank queries on a bit vector in constant time.
///
/// A rank query counts the number of set or unset bits, respectively, from the beginning up to a given position.
/// The data structure uses a hierarchical scheme dividing the bit vector into \em blocks (of 64 bits) and
/// \em superblocks and precomputes a number of rank queries, storing them in a space efficient manner.
/// On the lowest level, it uses \c popcnt instructions.
///
/// Note that this data structure is \em static.
/// It maintains a pointer to the underlying bit vector and will become invalid if that bit vector is changed after construction.
class bit_rank {
private:
    using idx_t = uint32_t; // increase to uint64_t for bit vectors longer than 2^32

    /// \brief Counts the number of set bits in a 64-bit word.
    /// \param v the word in question
    static constexpr uint8_t rank1_u64(const uint64_t v) {
        return __builtin_popcountll(v);
    }

    /// \brief Counts the number of set bits in a 64-bit word up to a given position.
    /// \param v the word in question
    /// \param x the x-least significant bit up to which to count
    static constexpr uint8_t rank1_u64(const uint64_t v, const uint8_t x) {
        return __builtin_popcountll(v & (UINT64_MAX << (~x & 63ULL))); // 63 - x
    }

    /// \brief Computes the rounded-up integer quotient of two numbers.
    /// \param a the dividend
    /// \param b the divisor
    static constexpr size_t idiv_ceil(const size_t a, const size_t b) {
        return ((a + b) - 1ULL) / b;
    }

    static constexpr size_t SUPERBLOCK_WIDTH = 12;
    static constexpr size_t SUPERBLOCK_SIZE = 1ULL << SUPERBLOCK_WIDTH;
    static constexpr size_t BLOCKS_PER_SUPERBLOCK = SUPERBLOCK_SIZE / 64ULL;
    
    const std::vector<bool>* m_bv;

    uint64_t block64(const size_t i) const {
        // read the 64 bits... afaik, there is no better way of doing this, because std::vector<bool> does not provide a data() pointer
        uint64_t w = 0;
        const size_t start = i * 64ULL;
        for(size_t j = start; j < start + 64ULL; j++) {
            w = (w << 1) | (*m_bv)[j];
        }
        return w;
    }

    std::vector<uint16_t> m_blocks; // the template integer must at least fit integers of SUPERBLOCK_WIDTH bits
    std::vector<idx_t> m_superblocks;

public:

    /// \brief Constructs the rank data structure for the given bit vector.
    /// \param bv the bit vector
    bit_rank(const std::vector<bool>& bv) : m_bv(&bv) {
        const size_t n = m_bv->size();
        const size_t num_blocks = idiv_ceil(n, 64ULL);
        const size_t num_superblocks = idiv_ceil(n, SUPERBLOCK_SIZE);

        m_blocks.reserve(num_blocks);
        m_superblocks.reserve(num_superblocks);

        // construct
        {
            size_t rank_bv = 0; // 1-bits in whole BV
            size_t rank_sb = 0; // 1-bits in current superblock
            size_t cur_sb = 0;  // current superblock

            for(size_t j = 0; j < num_blocks; j++) {
                if(j % BLOCKS_PER_SUPERBLOCK == 0) {
                    // we reached a new superblock
                    m_superblocks.emplace_back(rank_bv);
                    ++cur_sb;
                    assert(m_superblocks[cur_sb-1] == rank_bv);
                    rank_sb = 0;
                }
                
                m_blocks.emplace_back(rank_sb);

                const auto rank_b = rank1_u64(block64(j));
                rank_sb += rank_b;
                rank_bv += rank_b;
            }
        }
    }

    /// \brief Constructs an empty, uninitialized rank data structure.
    inline bit_rank() {
    }

    bit_rank(const bit_rank& other) = default;
    bit_rank(bit_rank&& other) = default;
    bit_rank& operator=(const bit_rank& other) = default;
    bit_rank& operator=(bit_rank&& other) = default;

    /// \brief Counts the number of set bit (1-bits) from the beginning of the bit vector up to (and including) position \c x.
    /// \param x the position until which to count
    size_t rank1(const size_t x) const {
        const size_t r_sb = m_superblocks[x / SUPERBLOCK_SIZE];
        const size_t j   = x / 64ULL;
        const size_t r_b = m_blocks[j];
        
        return r_sb + r_b + rank1_u64(block64(j), x & 63ULL);
    }

    /// \brief Counts the number of set bits from the beginning of the bit vector up to (and including) position \c x.
    ///
    /// This is a convenience alias for \ref rank1.
    ///
    /// \param x the position until which to count
    inline size_t operator()(const size_t x) const {
        return rank1(x);
    }

    /// \brief Counts the number of unset bits (0-bits) from the beginning of the bit vector up to (and including) position \c x.
    /// \param x the position until which to count
    inline size_t rank0(const size_t x) const {
        return x + 1 - rank1(x);
    }
};
