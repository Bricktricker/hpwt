#include <distwt/apps/mpi_dd.hpp>
#include <distwt/apps/mpi_launcher.hpp>

int main(int argc, char* argv[]) {
   return mpi_launch<mpi_dd>(argc, argv);
}
