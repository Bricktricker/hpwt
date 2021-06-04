add_executable(hpwt_ppc main_ppc.cpp)
target_link_libraries(hpwt_ppc distwt ${MPI_LIBRARIES} ${TLX_LIBRARIES})

add_executable(hpwt_pps main_pps.cpp)
target_link_libraries(hpwt_pps distwt ${MPI_LIBRARIES} ${TLX_LIBRARIES})
