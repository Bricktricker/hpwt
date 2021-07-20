#pragma once

#include <tlx/math/div_ceil.hpp>
#include <pwm/util/debug_assert.hpp>
#include <pwm/arrays/span.hpp>

template<typename items_t, size_t N>
class uint64_pack_t {
private:
    static_assert(sizeof(items_t) == 8, "items_t must be 64 bits in size.");

public:
    static inline size_t required_bufsize(const size_t num_items) {
        return tlx::div_ceil(num_items, N);
    }

    static uint64_t pack_uint64(const items_t& items);
    static void unpack_uint64(uint64_t u64, items_t& items);

    template<typename src_t>
    static void pack(
        const src_t& src,
        const size_t src_offs,
        uint64_t* dst,
        const size_t num) {

        items_t buf;
        size_t count = 0;

        // write items into buffer
        for(size_t i = 0; i < num; i++) {
            buf[count++] = src[src_offs + i];
            if(count >= N) {
                // buffer full, write to destination
                *dst++ = pack_uint64(buf);
                count = 0;
            }
        }

        // remainder
        if(count > 0) {
            *dst++ = pack_uint64(buf);
        }
    }

    static void packBlocks(const uint64_t* src,
        const size_t src_offs,
        uint64_t* _dst,
        const size_t num) {

            span<uint64_t> dst(_dst, required_bufsize(num));

            src += src_offs / 64ULL;

            const size_t block_offs = (src_offs % 64ULL);
            const size_t inv_block_offs = 64ULL - block_offs;
            const auto mask = (1ULL << block_offs) - 1;

            size_t src_pos = src_offs % N;
            size_t dst_pos = 0;

            while(dst_pos < num) {
                dst[dst_pos / N] = src[src_pos / N] << block_offs;
                src_pos += inv_block_offs;
                dst_pos += inv_block_offs;

                if(dst_pos >= num) {
                    //TODO: clear last bits
                    //dst[dst_pos / N] &= ~((1ULL << inv_block_offs) - 1); // does not work
                    return;
                }

                dst[dst_pos / N] |= (src[src_pos / N] & (mask << inv_block_offs)) >> inv_block_offs;
                src_pos += block_offs;
                dst_pos += block_offs;
            }
        }

    template<typename dst_t>
    static void unpack(
        const uint64_t* src,
        dst_t& dst,
        const size_t dst_offs,
        const size_t num) {

        items_t buf;
        size_t count = N;

        // read items into buffer
        for(size_t i = 0; i < num; i++) {
            if(count >= N) {
                unpack_uint64(*src++, buf);
                count = 0;
            }

            dst[dst_offs + i] = buf[count++];
        }
    }
};