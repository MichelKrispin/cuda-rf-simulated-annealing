//
// Created by Michel Krispin in January, 2024.
//

#ifndef CUDA_RF_SIMULATED_ANNEALING_TEMPERATURESCHEDULE_H
#define CUDA_RF_SIMULATED_ANNEALING_TEMPERATURESCHEDULE_H

#include <cstdint>
#include <vector>
#include <random>
#include <cmath>

class TemperatureSchedule {
private:
    float expectation_delta_x() {
        const uint32_t m = 32; // Number of samples

        float average = 0.0f;

        std::vector<uint8_t> x(this->n);
        for (uint32_t i = 0; i < n; i++) {
            x[i] = binary_distribution(generator);
        }


        std::vector<float> h(this->n);
        // Initialize h
        for (uint32_t i = 0; i < n; i++) {
            for (uint32_t j = 0; j < n; j++) {
                if (j != i) {
                    h[i] += Q[j + n * i] * x[j];
                }
            }
            h[i] += (1 - x[i]) * Q[i + n * i];
        }

        // Then loop through random changes
        for (uint32_t count = 0; count < m; count++) {
            uint32_t accepted_idx = n_distribution(generator);
            float inner_sum = 0.0f;
            for (uint32_t i = 0; i < n; i++) {
                if (i != accepted_idx) {
                    h[i] -= Q[accepted_idx + n * i] * (1 - x[accepted_idx]);
                    inner_sum += abs(h[i]);
                }
            }
            average += inner_sum / static_cast<float>(n);
        }
        average /= static_cast<float>(m);

        return average;
    }

    void sample_temperatures() {
        const float p_start = 0.99;
        const float p_trans = 0.5;
        const float nu = 0.9;

        const float e_delta_x = 2 * expectation_delta_x();
        t_0 = -e_delta_x / (logf(1 - powf(1 - p_start, 1.0f / static_cast<float>(n))));
        t_end = powf(t_0, 1.0f - (1.0f / nu)) * powf(
                -e_delta_x / (log(1 - powf(1 - p_trans, 1.0f / static_cast<float>(n)))),
                -1.0f / nu
        );
    }

    void create_temperature_schedule(const uint32_t &number_iterations) {
        sample_temperatures();

        double epsilon = exp(log(t_end / t_0) / static_cast<double>(number_iterations));
        for (uint32_t i = 0; i < number_iterations; i++) {
            //double t = 1.0 / (t_0 * pow(epsilon, static_cast<double>(i)));
            double t = t_0 * pow(epsilon, static_cast<double>(i));
            ts.push_back(static_cast<float>(t));
        }
    }

public:
    TemperatureSchedule(const uint32_t &number_iterations, const float *Q, uint32_t n) : Q(Q), n(n) {
        generator = std::mt19937(std::random_device()());
        binary_distribution = std::uniform_int_distribution<>(0, 1);
        n_distribution = std::uniform_int_distribution<>(0, n);

        create_temperature_schedule(number_iterations);
    }

    [[nodiscard]] const float *get() const {
        return this->ts.data();
    }

private:
    const float *Q;
    const uint32_t n;
    float t_0{}, t_end{};
    std::vector<float> ts;
    std::mt19937 generator;
    std::uniform_int_distribution<> binary_distribution;
    std::uniform_int_distribution<> n_distribution;
};

#endif //CUDA_RF_SIMULATED_ANNEALING_TEMPERATURESCHEDULE_H
