add_library(distwt
    common/bitrev.cpp
    common/result.cpp
    mpi/context.cpp
    mpi/malloc.cpp
    mpi/mpi_sum.cpp
    mpi/mpi_type.cpp
    mpi/uint64_pack.cpp
    mpi/wt_levelwise.cpp
)
target_link_libraries(distwt ${MPI_LIBRARIES})