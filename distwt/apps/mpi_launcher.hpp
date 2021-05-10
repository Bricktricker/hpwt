#pragma once

#include <tlx/cmdline_parser.hpp>
#include <distwt/mpi/context.hpp>

#include <distwt/mpi/uint_types.hpp>
#include <src/validate_distwt.hpp>

template<typename mpi_app_t>
int mpi_launch(int argc, char** argv) {
    // Read command-line
    tlx::CmdlineParser cp;

    size_t rdbufsize = 0; // default to local input size
    cp.add_bytes('r', "rbuf", rdbufsize, "File read buffer size.");

    std::string output("");
    cp.add_string('o', "output", output, "Name of output file.");

    size_t prefix = SIZE_MAX; // default to whole file
    cp.add_bytes('p', "prefix", prefix, "Only process prefix of input file.");

    size_t sym_width = 1;
    cp.add_bytes('w', "width", sym_width, "Number of bytes per input symbol.");

    bool eff_input = false;
    cp.add_flag('e', "effective", eff_input,
        "Input is already an effective transform (skip histogram computation).");

    bool validate_tree = false;
    cp.add_flag('v', "validate", validate_tree, "Validate the generated wavelet tree.");

    std::string input_filename; // required
    cp.add_param_string("file", input_filename, "The input file.");
    if (!cp.process(argc, argv)) {
        return -1;
    }

    // Validation requires that we write the tree to the filesystem
    validate_tree &= !output.empty();

    // Init MPI
    MPIContext ctx(&argc, &argv);

    // start
    switch(sym_width) {
        case 1:
            mpi_app_t::template start<uint8_t>(
                ctx,
                input_filename, prefix, rdbufsize, eff_input,
                output);
            if(validate_tree && ctx.is_master()) {
                validate_distwt<uint8_t>(input_filename, output, ctx.num_workers());
            }
            return 0;

        case 2:
            mpi_app_t::template start<uint16_t>(
                ctx,
                input_filename, prefix, rdbufsize, eff_input,
                output);
            if(validate_tree && ctx.is_master()) {
                validate_distwt<uint16_t>(input_filename, output, ctx.num_workers());
            }
            return 0;

        case 4:
            mpi_app_t::template start<uint32_t>(
                ctx,
                input_filename, prefix, rdbufsize, eff_input,
                output);
            if(validate_tree && ctx.is_master()) {
                validate_distwt<uint32_t>(input_filename, output, ctx.num_workers());
            }
            return 0;

        case 5:
            mpi_app_t::template start<uint40_t>(
                ctx,
                input_filename, prefix, rdbufsize, eff_input,
                output);
            if(validate_tree && ctx.is_master()) {
                ctx.cout_master() << "can not validate tree for 5 byte input symbol width\n";
            }
            return 0;

        default:
            ctx.cout_master()
                << "symbol width of " << sym_width << " not supported"
                << std::endl;
            return -2;
    }
}
