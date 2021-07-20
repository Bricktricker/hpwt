#pragma once

#include <cstddef>
#include <memory>
#include <new>
#include <pwm/util/common.hpp>

template <class T>
class Alignment_allocator {
    constexpr static size_t ALIGNMENT = CACHELINE_SIZE;

  public:
    using value_type = T;

    Alignment_allocator() noexcept {}
    template <class U>
    Alignment_allocator(Alignment_allocator<U> const&) noexcept {}

    value_type* allocate(std::size_t n) const {
        const size_t num_bytes = (n * sizeof(value_type)) + ALIGNMENT;
        const void* ptr = std::malloc(num_bytes);

        value_type* const res = reinterpret_cast<value_type*>((reinterpret_cast<std::uintptr_t>(ptr) & ~(size_t(ALIGNMENT - 1))) + ALIGNMENT);
        *(reinterpret_cast<const void**>(res) - 1) = ptr;
        
        return res;
    }

    void deallocate(value_type* p, std::size_t) const noexcept {
        if (p != nullptr) {
            const auto ptr = *(reinterpret_cast<void**>(p) - 1);
            std::free(ptr);
        }
    }
};

template <class T, class U>
bool operator==(Alignment_allocator<T> const&, Alignment_allocator<U> const&) noexcept {
    return true;
}

template <class T, class U>
bool operator!=(Alignment_allocator<T> const& x, Alignment_allocator<U> const& y) noexcept {
    return !(x == y);
}