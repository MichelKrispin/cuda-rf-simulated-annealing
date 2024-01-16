# CUDA Rejection-free Simulated Annealing

A CUDA implementation of rejection-free simulated annealing.
The algorithm is implemented close to the [python version](https://github.com/MichelKrispin/rf-simulated-annealing) and is therefore only a starting point for further optimizations.
Currently, the QUBO size is limited by the number of parallel threads that the GPU allows (1024 in my case).

[CUDA](https://developer.nvidia.com/cuda-downloads) must be installed to run the project.
Then, CMake is used as the build tool which means it can be built and run with
```shell
mkdir build
cmake -B build
cmake --build build
./build/cuda_rf_simulated_annealing qubo.npy 10000
```
The second argument is the number of simulated annealing iterations while the first argument is the QUBO matrix file.
The matrix file is a simple numpy file and if the problem is created in python (as an upper triangular matrix), the matrix can be saved using `np.save('qubo.npy', Q)`.
Or, as an example
```python
n = 512
np.save("Q.npy", np.triu(np.random.uniform(-4, 8, size=(n, n))))
```
For loading the numpy library [libnpy](https://github.com/llohse/libnpy/) is used.
