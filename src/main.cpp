#include "validate_distwt.hpp"
#include <bitset>
#include <distwt/apps/mpi_dd.hpp>
#include <distwt/mpi/context.hpp>
#include <distwt/mpi/uint_types.hpp>
#include <fstream>
#include <iostream>
#include <iterator>
#include <mpi.h>
#include <string>
#include <vector>

void computeDist(int argc, char* argv[]) {
    MPIContext ctx(&argc, &argv);
    const std::string output("output/out");
    const std::string input("test_input.txt");
    mpi_dd::template start<uint8_t>(ctx, input, SIZE_MAX /* prefix */, 0 /* rdbufsize */,
                                    false /* effective input (unused) */, output);
    if (ctx.is_master()) {
        validate_distwt<uint8_t>(input, output, ctx.num_workers());
    }
}

int main(int argc, char* argv[]) {
    computeDist(argc, argv);
    return EXIT_SUCCESS;
}
