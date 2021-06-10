#pragma once

#include <distwt/mpi/wt.hpp>
#include <distwt/mpi/context.hpp>

class WaveletTreeLevelwise : public WaveletTree {
public:
    template<typename sym_t>
    inline WaveletTreeLevelwise(
        const Histogram<sym_t>& hist,
        ctor_t construction_algorithm)
        : WaveletTree(hist, construction_algorithm) {
    }

    size_t save(const MPIContext& ctx, const std::string& output);
};