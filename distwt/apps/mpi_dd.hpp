#include <string>
#include <vector>

#include <distwt/common/util.hpp>
#include <distwt/common/wt_sequential.hpp>

#include <distwt/mpi/file_partition_reader.hpp>

#include <distwt/mpi/histogram.hpp>
#include <distwt/mpi/effective_alphabet.hpp>
#include <distwt/mpi/bit_vector.hpp>
#include <distwt/mpi/wt_nodebased.hpp>
#include <distwt/mpi/wt_levelwise.hpp>

#include <distwt/mpi/result.hpp>

class mpi_dd {
public:

template<typename sym_t>
static void start(
    MPIContext& ctx,
    const std::string& input_filename,
    const size_t prefix,
    const size_t in_rdbufsize,
    const bool eff_input,
    const std::string& output) {

    Result::Time time;
    double t0 = ctx.time();

    auto dt = [&](){
        const double t = ctx.time();
        const double dt = t - t0;
        t0 = t;
        return dt;
    };

    // Determine input partition
    FilePartitionReader<sym_t> input(ctx, input_filename, prefix);
    const size_t local_num = input.local_num();
    const size_t rdbufsize = (in_rdbufsize > 0) ? in_rdbufsize : local_num;
    input.buffer(rdbufsize);
    
    time.input = dt();

    // Compute histogram
    ctx.cout_master() << "Compute histogram ..." << std::endl;
    Histogram<sym_t> hist(ctx, input, rdbufsize);

    time.hist = dt();

    // Compute effective alphabet
    EffectiveAlphabet<sym_t> ea(hist);

    // Transform text and cache in RAM
    ctx.cout_master() << "Compute effective transformation ..." << std::endl;
    std::vector<sym_t> etext(local_num);
    {
        size_t i = 0;
        ea.transform(input, [&](sym_t x){ etext[i++] = x; }, rdbufsize);
    }

    time.eff = dt();
    input.free();

    // recursive WT
    ctx.cout_master() << "Compute local WTs ..." << std::endl;
    auto wt_nodes = WaveletTreeNodebased(hist,
    [&](WaveletTree::bits_t& bits, const WaveletTreeBase& wt){

        bits.resize(wt.num_nodes());
        wt_pc<sym_t, idx_t>(wt, bits, etext);
    });

    // Clean up
    etext.clear();
    etext.shrink_to_fit();

    // Synchronize
    ctx.cout_master() << "Done computing " << wt_nodes.num_nodes()
        << " nodes. Synchronizing ..." << std::endl;
    ctx.synchronize();

    time.construct = dt();

    // Convert to level-wise representation
    WaveletTreeLevelwise wt = wt_nodes.merge(ctx, input, hist, true);
    time.merge = dt();

    // write to disk if needed
    if(output.length() > 0) {
        ctx.synchronize();
        ctx.cout_master() << "Writing WT to disk ..." << std::endl;

        if(ctx.rank() == 0) {
            hist.save(output + "." + WaveletTreeBase::histogram_extension());
        }

        wt.save(ctx, output);
    }

    // Synchronize for exit
    ctx.cout_master() << "Waiting for exit signals ..." << std::endl;
    ctx.synchronize();

    // gather stats
    Result result("mpi-dd", ctx, input, wt.sigma(), time);

    ctx.cout_master() << result.readable() << std::endl
                      << result.sqlplot() << std::endl;

    //validate output
    if(ctx.is_master() && input.total_size() > 0) {
        binary::FileReader file_reader(input_filename);
        const auto bits = wt.raw_bits();
        const size_t input_size = bits[0].size();
        
        for(size_t i = 0; i < input_size; i++) {
            sym_t value = 0;
            size_t idx = i;
            size_t level_begin = 0;
            size_t level_end = input_size;
            for(size_t level = 0; level < wt.height(); level++) {
                const auto& level_bits = bits[level];
                const size_t num_level_zeros = std::accumulate(
                        std::next(level_bits.begin(), level_begin),
                        std::next(level_bits.begin(), level_end),
                        0,
                        [](const size_t acc, const bool v) { return v ? acc : acc+1; }); //increment acc on false to count 0's

                const auto bit = level_bits[idx];
                value <<= 1;
                value |= bit;
                if(!bit) {
                    //go left
                    //count number of 0's from level_begin to idx
                    idx = std::accumulate(
                        std::next(level_bits.begin(), level_begin),
                        std::next(level_bits.begin(), idx),
                        0,
                        [](const size_t acc, const bool v) { return v ? acc : acc+1; }); //increment acc on false to count 0's

                    level_end = level_begin + num_level_zeros;
                }else{
                    //go right
                    //count number of 1's from level_begin to idx
                    idx = std::accumulate(
                        std::next(level_bits.begin(), level_begin),
                        std::next(level_bits.begin(), idx),
                        0);
                    
                    level_begin = level_begin + num_level_zeros;
                }
                idx += level_begin;
            }

            //read char from file
            const sym_t file_value = file_reader.read<sym_t>();
            if(file_value != value) {
                //decoded and file value mismatch
                ctx.cout_master() << "Error while decoding wavelet tree on position " << i
                    << ", expected " << static_cast<size_t>(file_value) << ", but got " << static_cast<size_t>(value) << '\n';
            }
        }

    }
}

};
