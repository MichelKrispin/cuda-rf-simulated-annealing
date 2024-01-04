#include <string>
#include <iostream>
#include <chrono>

#include <curand.h>
#include <curand_kernel.h>

#include "TemperatureSchedule.h"
#include "greedy_search.h"
#include "npy.hpp"

#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

__global__ void setup_kernel(curandState *state, const uint64_t seed) {
    const uint32_t idx = threadIdx.x;
    curand_init(seed + idx, idx, 0, &state[idx]);
}

__global__ void
rf_simulated_annealing_loop(const float *Q, float *h, uint8_t *x, float *criteria, const uint32_t n,
                            const float *ts,
                            const uint32_t N, curandState *state) {
    const uint32_t i = threadIdx.x;
    // Ignore indices that are larger than the problem size
    if (i < n) {
        for (uint32_t t = 0; t < N; t++) {
            const float deltaE_i = (float) -(1 - 2 * (1 - x[i])) * h[i];
            const float u_i = curand_uniform(&state[i]);
            criteria[i] = fmaxf(0.0f, deltaE_i) + ts[t] * __logf(-__logf(u_i));
            __syncthreads();

            // Compute the argmin
            uint16_t accepted_idx = 0;
            {
                float minimum = criteria[0];
                for (uint32_t j = 1; j < n; j++) {
                    if (criteria[j] < minimum) {
                        accepted_idx = j;
                        minimum = criteria[j];
                    }
                }
            }

            // Accept the value
            const uint8_t flipped_value = 1 - x[accepted_idx];
            __syncthreads();
            float dh_i;
            if (i == accepted_idx) {
                dh_i = 0.0f;
                x[accepted_idx] = flipped_value;
            } else {
                dh_i = Q[accepted_idx + n * i] * (float) (1 - 2 * flipped_value);
            }
            // Update the h
            h[i] -= dh_i;
        }
    }
}

float computeQ(const float *Q, const uint8_t *x, const uint32_t &n) {
    float energy = 0.0;
    for (uint32_t i = 0; i < n; i++) {
        for (uint32_t j = i; j < n; j++) {
            energy += Q[j + n * i] * x[i] * x[j];
        }
    }
    return energy;
}

std::vector<float> load_Q(const std::string &qubo_filename) {
    // Load up the QUBO matrix from the python file
    // It is expected to be in triangular matrix form
    npy::npy_data Q_data = npy::read_npy<double>(qubo_filename);
    std::vector<float> Q(Q_data.data.begin(), Q_data.data.end()); // Convert double to float data
    std::vector<unsigned long> shape = Q_data.shape;
    if (shape.size() != 2) {
        std::cerr << "The Q matrix must have a dimension of 2!\n";
        return std::vector<float>();
    } else if (shape[0] != shape[1]) {
        std::cerr << "The Q matrix must be a square matrix!\n";
        return std::vector<float>();
    }

    const uint32_t n = Q_data.shape[0];
    // Create a mirrored version, such that Q[i, j] = Q[j, i] for easier computation
    // (This is not equal to the symmetric QUBO matrix; we still use the triangular version for computation
    // but don't need to do checks)
    for (uint32_t i = 0; i < n; i++) {
        for (uint32_t j = 0; j < n; j++) {
            Q[i + j * n] = Q[j + i * n];
        }
    }

    return Q;
}

std::vector<uint8_t> rf_simulated_annealing(std::vector<float> &Q, const uint32_t &num_iterations) {
    const auto n = static_cast<uint32_t>(sqrt(Q.size()));
    // Temperature schedule
    TemperatureSchedule temperatureSchedule(num_iterations, Q.data(), n);
    const float *ts = temperatureSchedule.get();

    // Initial random x
    std::random_device rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937 generator(rd()); // Standard mersenne_twister_engine seeded with rd()
    std::uniform_int_distribution<> binary_distribution(0, 1);
    std::vector<uint8_t> x(n);
    for (uint32_t i = 0; i < n; i++) {
        x[i] = binary_distribution(generator);
    }

    // Initialize h
    std::vector<float> h(n);
    for (uint32_t i = 0; i < n; i++) {
        for (uint32_t j = 0; j < n; j++) {
            h[i] += Q[j + n * i] * x[j];
        }
        h[i] += (1 - x[i]) * Q[i + n * i];
    }

    // Copy data to CUDA memory
    float *d_ts;
    uint8_t *d_x;
    float *d_fx;
    float *d_h;
    float *d_criteria;
    float *d_Q;
    cudaMalloc(&d_ts, num_iterations * sizeof(float));
    cudaMalloc(&d_x, n * sizeof(uint8_t));
    cudaMalloc(&d_fx, sizeof(float));
    cudaMalloc(&d_h, n * sizeof(float));
    cudaMalloc(&d_criteria, n * sizeof(float));
    cudaMalloc(&d_Q, n * n * sizeof(float));
    cudaCheckErrors("Memory allocation");
    cudaMemcpy(d_ts, ts, num_iterations * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x.data(), n * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_h, h.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Q, Q.data(), n * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckErrors("Memory copy");

    // Define grid and thread dimensions
    // This is not really a smart way to create the block dimension
    const uint32_t gridDimension = 1;
    const uint32_t blockDimension = n;

    // Create CUDA random states
    curandState *d_state;
    cudaMalloc(&d_state, sizeof(curandState) * blockDimension);
    const uint64_t seed = std::chrono::duration_cast<std::chrono::nanoseconds>(
            (std::chrono::system_clock::now()).time_since_epoch()).count();
    setup_kernel<<<gridDimension, blockDimension>>>(d_state, seed);
    cudaCheckErrors("Curand initialization");

    rf_simulated_annealing_loop<<<gridDimension, blockDimension>>>(
            d_Q, d_h, d_x, d_criteria, n, d_ts,
            num_iterations, d_state); // Limit of grid * block must be <= 2048
    cudaDeviceSynchronize();
    cudaCheckErrors("Kernel fail");

    // Copy the latest x back to the host
    cudaMemcpy(x.data(), d_x, n * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h.data(), d_h, n * sizeof(float), cudaMemcpyDeviceToHost);

    // Free the CUDA memory
    cudaFree(d_ts);
    cudaFree(d_x);
    cudaFree(d_fx);
    cudaFree(d_h);
    cudaFree(d_criteria);
    cudaFree(d_Q);

    // Then do a greedy local search with the found x to find the best possible solutions
    // (The rejection-free variant might have flipped to a worse solution at the end)
    greedy_search(Q.data(), h.data(), x.data(), n);
    return x;
}

