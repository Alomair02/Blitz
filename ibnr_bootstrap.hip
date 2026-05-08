#include <hip/hip_runtime.h>
#include <rocrand/rocrand_kernel.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

// --- Dimensions ---
#define N_COMPANIES  146
#define N_LAGS       10
#define N_BOOT       10000
#define BLOCK_SIZE   256    // threads per block = bootstrap iterations per block

// Grid layout: (ceil(N_BOOT/BLOCK_SIZE), N_COMPANIES)
// One thread = one bootstrap iteration for one company
// All threads in a block share the same company's triangle via shared memory

#define HIP_CHECK(cmd) do { \
    hipError_t e = (cmd); \
    if (e != hipSuccess) { \
        fprintf(stderr, "HIP error %s:%d: %s\n", __FILE__, __LINE__, \
                hipGetErrorString(e)); exit(1); \
    } \
} while(0)

// -----------------------------------------------------------------------
// RNG init
// -----------------------------------------------------------------------
__global__ void init_rng(
    rocrand_state_xorwow* states,
    unsigned long long seed,
    int n_total
) {
    int tid = blockIdx.y * (gridDim.x * blockDim.x)
            + blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_total) return;
    rocrand_init(seed, tid, 0, &states[tid]);
}

// -----------------------------------------------------------------------
// Bootstrap chain ladder kernel
//
// Shared memory per block:
//   s_tri[10][10]  -- company triangle (read-only for all threads)
//   s_lf[9][9]     -- link ratios per lag (up to 9 per lag)
//   s_nf[9]        -- count of valid ratios per lag
//
// Each thread:
//   1. Waits for thread 0 to populate shared memory
//   2. Samples bootstrap development factors by resampling link ratios
//   3. Projects lower triangle from latest diagonal
//   4. Writes total IBNR to output array
// -----------------------------------------------------------------------
__global__ void bootstrap_chainladder(
    const float* __restrict__ triangles,  // (N_COMPANIES, N_LAGS, N_LAGS)
    rocrand_state_xorwow*     states,     // (N_COMPANIES * N_BOOT)
    float*                    ibnr_out    // (N_COMPANIES, N_BOOT)
) {
    int company = blockIdx.y;
    int boot_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (company >= N_COMPANIES || boot_id >= N_BOOT) return;

    // ---------------------------------------------------------------
    // Shared memory: triangle + link ratios (initialised by thread 0)
    // ---------------------------------------------------------------
    __shared__ float s_tri[N_LAGS][N_LAGS];    // 400 bytes
    __shared__ float s_lf[N_LAGS-1][N_LAGS];  // 360 bytes  (lag × acc_year)
    __shared__ int   s_nf[N_LAGS-1];           //  36 bytes  (count per lag)

    if (threadIdx.x == 0) {
        // Load this company's triangle
        const float* base = triangles + company * N_LAGS * N_LAGS;
        for (int ay = 0; ay < N_LAGS; ay++)
            for (int lag = 0; lag < N_LAGS; lag++)
                s_tri[ay][lag] = base[ay * N_LAGS + lag];

        // Compute individual link ratios for each lag transition k → k+1
        // Upper triangle: ay + (k+1) <= 9  →  ay <= 8-k
        for (int k = 0; k < N_LAGS-1; k++) {
            int n = 0;
            for (int ay = 0; ay <= 8-k; ay++) {
                float c_k  = s_tri[ay][k];
                float c_k1 = s_tri[ay][k+1];
                if (c_k > 0.0f && c_k1 > 0.0f)
                    s_lf[k][n++] = c_k1 / c_k;
            }
            s_nf[k] = n;
        }
    }
    __syncthreads();  // all threads wait until shared memory is ready

    // ---------------------------------------------------------------
    // Each thread: one independent bootstrap iteration
    // ---------------------------------------------------------------
    rocrand_state_xorwow state = states[company * N_BOOT + boot_id];

    // Step 1: Sample bootstrap development factors
    // For each lag k, resample n_k link ratios with replacement → take mean
    float f_boot[N_LAGS-1];
    for (int k = 0; k < N_LAGS-1; k++) {
        int n = s_nf[k];
        if (n == 0) { f_boot[k] = 1.0f; continue; }

        float sum = 0.0f;
        for (int s = 0; s < n; s++) {
            int idx = (int)(rocrand(&state) % (unsigned int)n);
            sum += s_lf[k][idx];
        }
        f_boot[k] = sum / (float)n;
    }

    // Step 2: Project lower triangle from latest diagonal, compute IBNR
    // Latest diagonal: cell (ay, lag) where ay + lag == 9 (0-indexed)
    // Accident year ay=0 is fully developed at lag 9 — no IBNR needed
    float total_ibnr = 0.0f;

    for (int ay = 1; ay < N_LAGS; ay++) {
        int lag_diag = N_LAGS - 1 - ay;       // latest known lag index for this ay
        float cum    = s_tri[ay][lag_diag];    // latest diagonal value

        // Apply bootstrap factors from lag_diag to ultimate (index 8 = lag9→lag10)
        for (int k = lag_diag; k < N_LAGS-1; k++)
            cum *= f_boot[k];

        // IBNR = projected ultimate - latest known cumulative
        total_ibnr += cum - s_tri[ay][lag_diag];
    }

    ibnr_out[company * N_BOOT + boot_id] = total_ibnr;
    states[company * N_BOOT + boot_id]   = state;   // write back RNG state
}

// -----------------------------------------------------------------------
// CPU: compute percentile from sorted array
// -----------------------------------------------------------------------
static int cmp_float(const void* a, const void* b) {
    float fa = *(float*)a, fb = *(float*)b;
    return (fa > fb) - (fa < fb);
}

static float pct(float* sorted, int n, float p) {
    int idx = (int)(p * n);
    return sorted[(idx < n) ? idx : n-1];
}

static double now_sec() {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main() {
    printf("GPU Bootstrap Chain Ladder — IBNR Reserve Distribution\n");
    printf("Companies: %d  |  Bootstrap samples: %d  |  Lags: %d\n\n",
           N_COMPANIES, N_BOOT, N_LAGS);

    // --- Load triangle data ---
    float* h_tri  = (float*)malloc(N_COMPANIES * N_LAGS * N_LAGS * sizeof(float));
    FILE* fp = fopen("triangles.bin", "rb");
    if (!fp) { fprintf(stderr, "Cannot open triangles.bin\n"); return 1; }
    fread(h_tri, sizeof(float), N_COMPANIES * N_LAGS * N_LAGS, fp);
    fclose(fp);
    printf("Loaded triangles.bin  (%.1fKB)\n",
           N_COMPANIES * N_LAGS * N_LAGS * 4.0 / 1024);

    // --- GPU allocations ---
    float*                d_tri;
    rocrand_state_xorwow* d_states;
    float*                d_ibnr;
    long n_states = (long)N_COMPANIES * N_BOOT;

    HIP_CHECK(hipMalloc(&d_tri,    N_COMPANIES * N_LAGS * N_LAGS * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_states, n_states * sizeof(rocrand_state_xorwow)));
    HIP_CHECK(hipMalloc(&d_ibnr,   n_states * sizeof(float)));

    HIP_CHECK(hipMemcpy(d_tri, h_tri,
                        N_COMPANIES * N_LAGS * N_LAGS * sizeof(float),
                        hipMemcpyHostToDevice));

    // --- Init RNG ---
    // Grid covers all (company, boot_id) pairs
    int blocks_per_co = (N_BOOT + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 rng_grid(blocks_per_co, N_COMPANIES);
    dim3 block(BLOCK_SIZE);

    printf("Initialising %ld RNG states...\n", n_states);
    hipLaunchKernelGGL(init_rng, rng_grid, block, 0, 0,
                       d_states, 99ULL, (int)n_states);
    HIP_CHECK(hipDeviceSynchronize());

    // --- Run bootstrap kernel ---
    printf("Running bootstrap (%d blocks × %d threads)...\n",
           blocks_per_co * N_COMPANIES, BLOCK_SIZE);
    double t0 = now_sec();

    hipLaunchKernelGGL(bootstrap_chainladder,
                       dim3(blocks_per_co, N_COMPANIES), block, 0, 0,
                       d_tri, d_states, d_ibnr);
    HIP_CHECK(hipDeviceSynchronize());

    double elapsed = now_sec() - t0;
    printf("Done in %.3fs  (%.1fM projections/s)\n\n",
           elapsed, (double)N_COMPANIES * N_BOOT / elapsed / 1e6);

    // --- Copy results to host ---
    float* h_ibnr = (float*)malloc(n_states * sizeof(float));
    HIP_CHECK(hipMemcpy(h_ibnr, d_ibnr, n_states * sizeof(float),
                        hipMemcpyDeviceToHost));

    // --- Load company names ---
    char company_names[N_COMPANIES][128];
    FILE* fc = fopen("companies.txt", "r");
    for (int i = 0; i < N_COMPANIES; i++) {
        if (fc && fgets(company_names[i], 128, fc)) {
            // strip newline
            char* nl = strchr(company_names[i], '\n');
            if (nl) *nl = '\0';
        } else {
            snprintf(company_names[i], 128, "Company %d", i);
        }
    }
    if (fc) fclose(fc);

    // --- Per-company statistics ---
    float* sorted = (float*)malloc(N_BOOT * sizeof(float));

    printf("%-40s  %8s  %8s  %8s  %8s  %8s  %6s\n",
           "Company", "Mean", "Std", "P50", "P75", "P95", "CV%");
    printf("%s\n", "------------------------------------------------------------------------"
                   "----------------------------");

    // Save summary CSV
    FILE* fo = fopen("ibnr_summary.csv", "w");
    fprintf(fo, "company_idx,grcode_name,mean_ibnr,std_ibnr,p50,p75,p95,p99,cv_pct\n");

    double grand_mean = 0;
    for (int co = 0; co < N_COMPANIES; co++) {
        float* samples = h_ibnr + (long)co * N_BOOT;

        // Copy and sort for percentiles
        memcpy(sorted, samples, N_BOOT * sizeof(float));
        qsort(sorted, N_BOOT, sizeof(float), cmp_float);

        double mean = 0, var = 0;
        for (int b = 0; b < N_BOOT; b++) mean += samples[b];
        mean /= N_BOOT;
        for (int b = 0; b < N_BOOT; b++) var += (samples[b]-mean)*(samples[b]-mean);
        var /= N_BOOT;
        double std  = sqrt(var);
        double cv   = (mean > 0) ? std/mean*100.0 : 0;

        float p50 = pct(sorted, N_BOOT, 0.50f);
        float p75 = pct(sorted, N_BOOT, 0.75f);
        float p95 = pct(sorted, N_BOOT, 0.95f);
        float p99 = pct(sorted, N_BOOT, 0.99f);

        grand_mean += mean;

        // Truncate company name for display
        char display[41];
        snprintf(display, 41, "%s", company_names[co]);

        printf("%-40s  %8.0f  %8.0f  %8.0f  %8.0f  %8.0f  %5.1f%%\n",
               display, mean, std, (double)p50, (double)p75, (double)p95, cv);

        fprintf(fo, "%d,%s,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n",
                co, company_names[co],
                mean, std, (double)p50, (double)p75, (double)p95, (double)p99, cv);
    }

    fclose(fo);
    printf("\n%-40s  %8.0f\n", "TOTAL (sum of means)", grand_mean);
    printf("\nSaved ibnr_summary.csv\n");

    // Save full bootstrap samples for Python analysis
    FILE* fb = fopen("ibnr_samples.bin", "wb");
    fwrite(h_ibnr, sizeof(float), n_states, fb);
    fclose(fb);
    printf("Saved ibnr_samples.bin  (%.1fMB)\n",
           n_states * 4.0 / 1e6);

    // Cleanup
    HIP_CHECK(hipFree(d_tri));
    HIP_CHECK(hipFree(d_states));
    HIP_CHECK(hipFree(d_ibnr));
    free(h_tri); free(h_ibnr); free(sorted);
    return 0;
}
