cmake_minimum_required(VERSION 3.22...3.27)
project(cuda_rf_simulated_annealing CUDA)

set(CMAKE_CUDA_STANDARD 17)

add_executable(cuda_rf_simulated_annealing
        main.cu
        npy.hpp
        TemperatureSchedule.h
        greedy_search.h)

set_target_properties(cuda_rf_simulated_annealing PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
