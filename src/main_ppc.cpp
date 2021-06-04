#include <distwt/apps/mpi_dd.hpp>
#include <distwt/apps/mpi_launcher.hpp>

#include <src/wt_ppc_nodebased.hpp>

int main(int argc, char* argv[]) {
   return mpi_launch<mpi_dd<wt_ppc_nodebased>>(argc, argv);
}