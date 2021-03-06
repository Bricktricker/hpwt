/*******************************************************************************
 * include/util/alphabet_util.hpp
 *
 * Copyright (C) 2017 Florian Kurpicz <florian.kurpicz@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-2 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include <limits>
#include <type_traits>
#include <unordered_map>
#include <vector>
#include <set>

template <typename AlphabetType>
static uint64_t reduce_alphabet(std::vector<AlphabetType>& text) {
  uint64_t max_char = uint64_t(0);
  if constexpr (std::is_same<AlphabetType, uint8_t>::value) {
    std::array<uint64_t, std::numeric_limits<uint8_t>::max()> occ;
    occ.fill(0);
    for (auto& c : text) {
      if (occ[c] == 0) {
        occ[c] = ++max_char;
      }
      c = occ[c] - 1;
    }
    --max_char;
  } else {
    std::unordered_map<AlphabetType, uint64_t> word_list;
    for (const AlphabetType& character : text) {
      auto result = word_list.find(character);
      if (result == word_list.end()) {
        word_list.emplace(character, max_char++);
      }
    }
    --max_char;
    for (uint64_t i = 0; i < text.size(); ++i) {
      text[i] = static_cast<AlphabetType>(word_list.find(text[i])->second);
    }
  }
  return max_char;
}

template <typename AlphabetType>
static uint64_t reduce_textbook(std::vector<AlphabetType>& text) {
  if(text.empty()) {
    return 0;
  }

  std::set<AlphabetType> presentChars(text.begin(), text.end());
  std::vector<AlphabetType> charOrder(presentChars.begin(), presentChars.end());
  std::unordered_map<AlphabetType, AlphabetType> word_list;
  for(size_t i = 0; i < charOrder.size(); i++) {
    word_list[charOrder[i]] = static_cast<AlphabetType>(i);
  }

  for (auto& c : text) {
    c = word_list.find(c)->second;
  }

  return word_list.find(charOrder.back())->second;
}

// removed stxxlvector implementation

[[maybe_unused]] static uint64_t levels_for_max_char(uint64_t max_char) {
  uint64_t levels = 0;
  while (max_char) {
    max_char >>= 1;
    ++levels;
  }
  return levels;
}

template <typename AlphabetType>
static uint64_t no_reduction_alphabet(const std::vector<AlphabetType>& /*t*/) {
  uint64_t max_char = std::numeric_limits<AlphabetType>::max();
  uint64_t levels = 0;
  while (max_char) {
    max_char >>= 1;
    ++levels;
  }
  return levels;
}

/******************************************************************************/
