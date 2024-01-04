//
// Created by Michel Krispin in January, 2024.
//

#ifndef CUDA_RF_SIMULATED_ANNEALING_GREEDY_SEARCH_H
#define CUDA_RF_SIMULATED_ANNEALING_GREEDY_SEARCH_H

#include <random>
#include <cstdint>
#include <cmath>

void greedy_search(const float *Q, float *h, uint8_t *x, const uint32_t n) {
    std::vector<float> deltaE_i(n);

    uint16_t last_accepted_flip = n+1;
    while(1) {
        for (uint32_t i = 0; i < n; i++) {
            deltaE_i[i] = (float) -(1 - 2 * (1 - x[i])) * h[i];
        }

        uint16_t accepted_idx;
        uint8_t flipped_value;
        for (uint32_t i = 0; i < n; i++) {
            accepted_idx = 0;
            float minimum = deltaE_i[0];
            for (uint32_t j = 1; j < n; j++) {
                if (deltaE_i[j] < minimum) {
                    accepted_idx = j;
                    minimum = deltaE_i[j];
                }
            }
            flipped_value = 1 - x[accepted_idx];
        }

        // If nothing changed, quit
        if (accepted_idx == last_accepted_flip) {
            break;
        }
        last_accepted_flip = accepted_idx;

        for (uint32_t i = 0; i < n; i++) {
            // Accept the value
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


#endif //CUDA_RF_SIMULATED_ANNEALING_GREEDY_SEARCH_H
