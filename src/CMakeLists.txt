add_executable(hpwt
    main.cpp
)
target_link_libraries(hpwt distwt ${MPI_LIBRARIES} ${TLX_LIBRARIES})