#include <iostream>
#include <string>
#include <cstdlib>

#include "rf_simulated_annealing.cu"

int main(int argc, char **argv) {
    std::string qubo_filename("Q.npy");
    uint32_t num_iterations = 10000;
    const std::string usage("Usage: ./cuda_rf_simulated_annealing Q.npy 10000\n\n");
    if (argc < 2) {
        std::cout << "No arguments given. Using \"Q.npy\" as the qubo file and 10,000 iterations.\n"
                  << usage;
    } else if (argc > 3) {
        std::cerr << "Too many arguments given!\n"
                  << usage;
        return -1;
    } else if (argc != 3) {
        std::cerr << "Invalid number of arguments given!\n"
                  << usage;
        return -1;
    } else {
        qubo_filename = std::string(argv[1]);
        num_iterations = atoi(argv[2]);
    }

    auto Q = load_Q(qubo_filename);
    auto start_time = std::chrono::system_clock::now();
    auto x = rf_simulated_annealing(Q, num_iterations);
    auto end_time = std::chrono::system_clock::now();
    std::cout << "Computation took " << static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time).count()) / 1000.0 << " seconds\n";

    const uint32_t n = x.size();
    const float best_energy = computeQ(Q.data(), x.data(), n);

    std::printf("Computation finished with a found energy of %f\n", best_energy);
    std::cout << "x: [";
    for (uint32_t i = 0; i < n - 1; i++) {
        std::cout << (uint32_t) x[i] << ", ";
    }
    std::cout << (uint32_t) x[n - 1] << "]\n";
    return 0;
}
