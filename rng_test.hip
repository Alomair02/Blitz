#include <hip/hip_runtime.h>
#include <rocrand/rocrand_kernel.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define HIP_CHECK(cmd) do { \
    hipError_t e = (cmd); \
    if (e != hipSuccess) { \
        fprintf(stderr, "HIP error %s:%d -- %s\n", __FILE__, __LINE__, hipGetErrorString(e)); \
        exit(1); \
    } \
} while(0)

// Number of independent random streams (one per thread)
#define N_THREADS 65536
// Samples each thread generates
#define N_SAMPLES 100

// Each thread owns one rocrand state — its private RNG stream
__global__ void init_rng(rocrand_state_xorwow* states, unsigned long long seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N_THREADS) return;

    // Each thread gets the same seed but a different sequence number
    // This guarantees statistically independent streams
    rocrand_init(seed, tid, 0, &states[tid]);
}

__global__ void sample_normals(
    rocrand_state_xorwow* states,
    float* means,      // output: per-thread sample mean
    float* variances   // output: per-thread sample variance
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N_THREADS) return;

    rocrand_state_xorwow local_state = states[tid]; // copy to registers for speed

    float sum = 0.0f;
    float sum_sq = 0.0f;

    for (int i = 0; i < N_SAMPLES; i++) {
        // rocrand_normal() gives a N(0,1) float using Box-Muller internally
        float z = rocrand_normal(&local_state);
        sum    += z;
        sum_sq += z * z;
    }

    means[tid]     = sum / N_SAMPLES;
    variances[tid] = (sum_sq / N_SAMPLES) - (means[tid] * means[tid]);

    states[tid] = local_state; // write state back if we need to continue later
}

int main() {
    // --- Allocate device memory ---
    rocrand_state_xorwow* d_states;
    float* d_means;
    float* d_variances;

    HIP_CHECK(hipMalloc(&d_states,    N_THREADS * sizeof(rocrand_state_xorwow)));
    HIP_CHECK(hipMalloc(&d_means,     N_THREADS * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_variances, N_THREADS * sizeof(float)));

    int block_size = 256;
    int n_blocks   = (N_THREADS + block_size - 1) / block_size;

    // --- Initialize RNG states ---
    hipLaunchKernelGGL(init_rng, dim3(n_blocks), dim3(block_size), 0, 0,
                       d_states, 12345ULL);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    // --- Generate samples ---
    hipLaunchKernelGGL(sample_normals, dim3(n_blocks), dim3(block_size), 0, 0,
                       d_states, d_means, d_variances);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    // --- Copy results back ---
    float* h_means     = new float[N_THREADS];
    float* h_variances = new float[N_THREADS];
    HIP_CHECK(hipMemcpy(h_means,     d_means,     N_THREADS * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_variances, d_variances, N_THREADS * sizeof(float), hipMemcpyDeviceToHost));

    // --- Aggregate across all threads ---
    double grand_mean = 0.0, grand_var = 0.0;
    for (int i = 0; i < N_THREADS; i++) {
        grand_mean += h_means[i];
        grand_var  += h_variances[i];
    }
    grand_mean /= N_THREADS;
    grand_var  /= N_THREADS;

    printf("Samples generated : %d  (%d threads x %d samples)\n",
           N_THREADS * N_SAMPLES, N_THREADS, N_SAMPLES);
    printf("Grand mean        : %+.6f  (expected ~0.0)\n", grand_mean);
    printf("Grand variance    : %+.6f  (expected ~1.0)\n", grand_var);
    printf("Grand std dev     : %+.6f  (expected ~1.0)\n", sqrt(grand_var));

    // --- Cleanup ---
    HIP_CHECK(hipFree(d_states));
    HIP_CHECK(hipFree(d_means));
    HIP_CHECK(hipFree(d_variances));
    delete[] h_means;
    delete[] h_variances;

    return 0;
}
