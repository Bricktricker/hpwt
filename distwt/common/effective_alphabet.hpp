#pragma once

#include <unordered_map>

#include <distwt/common/histogram.hpp>

template<typename sym_t>
class EffectiveAlphabetBase {
protected:
    std::unordered_map<sym_t, sym_t> m_map;

public:
    template<typename idx_t>
    inline EffectiveAlphabetBase(const HistogramBase<sym_t, idx_t>& hist) {
        size_t i = 0;
        for(auto e : hist.entries) {
            m_map.emplace(e.first, sym_t(i++));
        }
    }

    inline ~EffectiveAlphabetBase() {
    }

    inline sym_t map(const sym_t value) const {
        return m_map.at(value);
    }
};