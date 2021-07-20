#pragma once
#include <cstdint>
#include <ostream>
#include <pwm/util/debug_assert.hpp>
#include <tlx/math/div_ceil.hpp>
#include <vector>

class bit_vector {
    constexpr static size_t BLOCK_SIZE = 64ULL;

    constexpr size_t get_block(const size_t idx) const noexcept {
        return idx / BLOCK_SIZE;
    }

  public:
    bit_vector() : m_num_bits(0U){};
    bit_vector(const size_t num_bits)
        : m_buffer(tlx::div_ceil(num_bits, BLOCK_SIZE), 0U), m_num_bits(0U) {}

    [[nodiscard]] bool get(const size_t idx) const {
        const size_t block = get_block(idx);
        DCHECK_LT(block, num_blocks());
        const size_t bit_idx = BLOCK_SIZE - (idx % BLOCK_SIZE) - 1;
        return (m_buffer[block] & (1ULL << bit_idx)) != 0;
    }

    [[nodiscard]] bool operator[](const size_t idx) const {
        return get(idx);
    }

    void set(const size_t idx, const bool val) {
        const size_t block = get_block(idx);
        DCHECK_LT(block, num_blocks());
        const size_t bit_idx = BLOCK_SIZE - (idx % BLOCK_SIZE) - 1;
        auto block_val = m_buffer[block] & (~(1ULL << bit_idx));
        block_val |= static_cast<uint64_t>(val) << bit_idx;
        m_buffer[block] = block_val;
    }

    void push_back(const bool val) {
        const size_t block = get_block(m_num_bits + 1);
        if (block >= m_buffer.size()) {
            m_buffer.push_back(0U);
        }
        set(m_num_bits, val);
        m_num_bits++;
    }

    [[nodiscard]] size_t size() const noexcept {
        return m_num_bits;
    }

    [[nodiscard]] bool empty() const noexcept {
        return m_num_bits == 0;
    }

    [[nodiscard]] size_t num_blocks() const noexcept {
        return m_buffer.size();
    }

    [[nodiscard]] const uint64_t* data() const noexcept {
        return m_buffer.data();
    }

    [[nodiscard]] uint64_t* data() noexcept {
        return m_buffer.data();
    }

    void set_block(const size_t block_idx, const uint64_t val) noexcept {
        DCHECK_LT(block_idx, num_blocks());
        m_buffer[block_idx] = val;
    }

    void resize(const size_t num_bits) {
        m_buffer.resize(tlx::div_ceil(num_bits, BLOCK_SIZE));
        m_num_bits = num_bits;
    }

    void clear() {
        m_buffer.clear();
        m_buffer.shrink_to_fit();
        m_num_bits = 0U;
    }

    friend std::ostream& operator<<(std::ostream& os, const bit_vector& vec) {
        for (size_t i = 0; i < vec.size(); i++) {
            const char c = (vec.get(i) ? '1' : '0');
            os << c;
        }
        return os;
    }

    bool operator==(const bit_vector& other) const {
        const auto size_this = size();
        const auto size_other = other.size();
        if (size_this != size_other) {
            return false;
        }
        for (size_t i = 0; i < size_this; i++) {
            const bool bit_this = get(i);
            const bool bit_other = other.get(i);
            if (bit_this != bit_other) {
                return false;
            }
        }
        return true;
    }

    bool operator!=(const bit_vector& rhs) const {
        return !operator==(rhs);
    }

  private:
    std::vector<uint64_t> m_buffer;
    size_t m_num_bits;
};