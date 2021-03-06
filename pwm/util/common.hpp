/*******************************************************************************
 * include/util/common.hpp
 *
 * Copyright (C) 2017 Florian Kurpicz <florian.kurpicz@tu-dortmund.de>
 * Copyright (C) 2017 Marvin Löbel <loebel.marvin@gmail.com>
 *
 * All rights reserved. Published under the BSD-2 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include <new>
#include <climits>
#include <stdint.h>
#include <type_traits>
#include <vector>
#include <pwm/util/debug_assert.hpp>

#include <pwm/util/permutation.hpp>

constexpr uint64_t word_size(uint64_t size) {
  return (size + 63ULL) >> 6;
}

// A little template helper for dropping a type early
template <typename T>
void drop_me(T const&) = delete;
template <typename T>
void drop_me(T&) = delete;
template <typename T>
void drop_me(T const&&) = delete;
template <typename T>
void drop_me(T&& t) {
  std::remove_reference_t<T>(std::move(t));
}

constexpr uint64_t log2(uint64_t n) {
  return (n < 2) ? 1 : 1 + log2(n / 2);
}

constexpr uint64_t pwmlog2(uint64_t n) {
  return (n < 2) ? 1 : 1 + log2(n / 2);
}

template <typename T>
inline static T mul64(const T t) {
  return t << 6;
}

template <typename T>
inline static T div64(const T t) {
  return t >> 6;
}

template <typename T>
inline static T mod64(const T t) {
  return t - mul64(div64(t));
}

template <typename WordType = uint64_t, typename bv_t>
inline auto bit_at(const bv_t& bv, uint64_t i) -> bool {
  constexpr WordType BITS = (sizeof(WordType) * CHAR_BIT);
  constexpr WordType MOD_MASK = BITS - 1;

  const uint64_t offset = i / BITS;
  const uint64_t word_offset = i & MOD_MASK;
  return (bv[offset] >> (MOD_MASK - word_offset)) & 1ULL;
}

// For some reason no current C++ compiler supported this C++17 feature in 2019-03:
// constexpr uint64_t CACHELINE_SIZE = std::hardware_destructive_interference_size;
// So we just hardcode a suitable value here.
// A value of 64 would be enough for current hardware generations,
// but we pick something that is a bit more future compatible.

constexpr uint64_t CACHELINE_SIZE = 128;

/******************************************************************************/
