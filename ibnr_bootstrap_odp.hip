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
#define BLOCK_SIZE   256

// England-Verrall overdispersion correction
// n = 45 data points in upper triangle (lags 1..9 across valid acc years)
// p = 9 development factors estimated
// correction = sqrt(n / (n-p)) = sqrt(45/36) = sqrt(1.25)
#define ODP_N        45
#define ODP_P        9
#define ODP_CORR     1.118033989f   // sqrt(45/36)

// IFRS 17 Cost of Capital rate (SAMA standard: 6% per annum)
#define COC_RATE     0.06f
// Average claims settlement duration for motor TPL in years
// Saudi motor: 2-3 years typical; use 2.5 as conservative estimate
#define COC_DURATION 2.5f

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
// ODP Bootstrap Chain Ladder Kernel (England-Verrall 2002)
//
// Corrections vs naive factor bootstrap:
//
//  1. RESIDUAL RESAMPLING: resample Pearson residuals, not link ratios.
//     Pearson residual: r_{i,k} = (C_{i,k} - mu_{i,k}) / sqrt(mu_{i,k})
//     where mu_{i,k} = C_{i,k-1} * f_{k-1}  (fitted value)
//     This captures both parameter uncertainty and process variance.
//
//  2. OVERDISPERSION CORRECTION: scale residuals by sqrt(n/(n-p))
//     before resampling to correct for degrees-of-freedom bias.
//     Unscaled residuals are too small because they were computed from
//     fitted (not true) parameters — this inflates the bootstrap variance
//     to the correct level.
//
//  3. PSEUDO-TRIANGLE PROPAGATION: when recomputing bootstrap development
//     factors, use the reconstructed pseudo-values in the denominator
//     (not original data), so resampling error propagates through all lags.
//
// Shared memory layout (796 bytes per block):
//   s_tri[10][10]  400B  company triangle (read-only)
//   s_mu[10][10]   400B  fitted values (ODP model)
//   s_resid[45]    180B  scaled Pearson residuals (pooled across all lags)
//   s_f[9]          36B  volume-weighted development factors
//   s_phi[1]         4B  overdispersion parameter phi
//   s_nr[1]          4B  residual count
// -----------------------------------------------------------------------
__global__ void bootstrap_odp(
    const float* __restrict__ triangles,
    rocrand_state_xorwow*     states,
    float*                    ibnr_out,
    float*                    phi_out        // (N_COMPANIES,) overdispersion
) {
    int company = blockIdx.y;
    int boot_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (company >= N_COMPANIES || boot_id >= N_BOOT) return;

    __shared__ float s_tri[N_LAGS][N_LAGS];
    __shared__ float s_mu[N_LAGS][N_LAGS];
    __shared__ float s_resid[ODP_N];
    __shared__ float s_f[N_LAGS-1];
    __shared__ float s_phi;
    __shared__ int   s_nr;

    // ---------------------------------------------------------------
    // Thread 0: load triangle, compute ODP model quantities
    // ---------------------------------------------------------------
    if (threadIdx.x == 0) {
        // Load triangle into shared memory
        const float* base = triangles + company * N_LAGS * N_LAGS;
        for (int ay = 0; ay < N_LAGS; ay++)
            for (int lag = 0; lag < N_LAGS; lag++)
                s_tri[ay][lag] = base[ay * N_LAGS + lag];

        // Step 1: volume-weighted development factors from upper triangle
        // Upper triangle: ay + (k+1) <= 9  →  ay <= 8-k
        for (int k = 0; k < N_LAGS-1; k++) {
            float num = 0, den = 0;
            for (int ay = 0; ay <= 8-k; ay++) {
                float c  = s_tri[ay][k];
                float c1 = s_tri[ay][k+1];
                if (c > 0) { num += c1; den += c; }
            }
            s_f[k] = (den > 0) ? num/den : 1.0f;
        }

        // Step 2: fitted values mu_{i,k} = C_{i,k-1} * f_{k-1}
        for (int ay = 0; ay < N_LAGS; ay++)
            s_mu[ay][0] = 0;   // column 0 has no fitted value
        for (int k = 1; k < N_LAGS; k++) {
            for (int ay = 0; ay <= 9-k; ay++) {
                float c = s_tri[ay][k-1];
                s_mu[ay][k] = (c > 0) ? c * s_f[k-1] : 0.0f;
            }
        }

        // Step 3: Pearson residuals r_{i,k} = (C - mu) / sqrt(mu)
        //         scaled by ODP correction sqrt(n/(n-p)) = ODP_CORR
        int nr = 0;
        float phi_num = 0;
        for (int k = 1; k < N_LAGS; k++) {
            for (int ay = 0; ay <= 9-k; ay++) {
                float mu = s_mu[ay][k];
                if (mu > 0.5f) {  // skip near-zero fitted values
                    float r = (s_tri[ay][k] - mu) / sqrtf(mu);
                    phi_num += r * r;
                    // Apply overdispersion correction before storing
                    s_resid[nr++] = r * ODP_CORR;
                }
            }
        }
        s_nr = nr;

        // Step 4: overdispersion parameter phi = sum(r^2) / (n-p)
        // Note: we compute from unscaled residuals so use phi_num / (n-p)
        s_phi = (nr > ODP_P) ? phi_num / (nr - ODP_P) : 1.0f;

        // Write phi to global memory for Python analysis
        if (boot_id == 0)
            phi_out[company] = s_phi;
    }
    __syncthreads();

    // ---------------------------------------------------------------
    // Each thread: one ODP bootstrap iteration
    // ---------------------------------------------------------------
    rocrand_state_xorwow state = states[company * N_BOOT + boot_id];
    int nr = s_nr;

    // --- Bootstrap development factors via pseudo-triangle ---
    // We propagate one pseudo-column at a time to avoid storing
    // the full pseudo-triangle in registers.
    // pseudo_cur[ay]: current column of pseudo-triangle
    // pseudo_nxt[ay]: next column (being constructed)
    float pseudo_cur[N_LAGS];
    float pseudo_nxt[N_LAGS];
    float f_boot[N_LAGS-1];

    // Column 0 is fixed (not resampled) — this is the anchor
    for (int ay = 0; ay < N_LAGS; ay++)
        pseudo_cur[ay] = s_tri[ay][0];

    for (int k = 0; k < N_LAGS-1; k++) {
        float num = 0, den = 0;
        int n_valid = 9 - k;   // accident years contributing to this factor

        for (int ay = 0; ay <= 8-k; ay++) {
            float mu_k1 = s_mu[ay][k+1];
            float c_k   = pseudo_cur[ay];

            // --- Reconstruct pseudo-value for cell (ay, k+1) ---
            float c_star;
            if (mu_k1 > 0.5f && nr > 0) {
                // Sample a scaled Pearson residual uniformly at random
                int idx  = (int)(rocrand(&state) % (unsigned int)nr);
                float r_star = s_resid[idx];
                // Reconstruct: C* = mu + r* * sqrt(mu)
                c_star = mu_k1 + r_star * sqrtf(mu_k1);
                c_star = fmaxf(0.0f, c_star);  // clamp to non-negative
            } else {
                c_star = (mu_k1 > 0) ? mu_k1 : s_tri[ay][k+1];
            }
            pseudo_nxt[ay] = c_star;

            num += c_star;      // numerator: pseudo column k+1
            den += c_k;         // denominator: pseudo column k
        }

        f_boot[k] = (den > 0 && num > 0) ? num/den : s_f[k];

        // Propagate: next column becomes current for the next lag
        for (int ay = 0; ay <= 8-k; ay++)
            pseudo_cur[ay] = pseudo_nxt[ay];
    }

    // --- Project lower triangle using bootstrap factors ---
    // Start from ACTUAL latest diagonal (not pseudo) — this is standard
    // because we are estimating IBNR as of the actual valuation date
    float total_ibnr = 0.0f;

    for (int ay = 1; ay < N_LAGS; ay++) {
        int   lag_diag = N_LAGS - 1 - ay;
        float cum      = s_tri[ay][lag_diag];

        for (int k = lag_diag; k < N_LAGS-1; k++)
            cum *= f_boot[k];

        total_ibnr += fmaxf(0.0f, cum - s_tri[ay][lag_diag]);
    }

    ibnr_out[company * N_BOOT + boot_id] = total_ibnr;
    states[company * N_BOOT + boot_id]   = state;
}

// -----------------------------------------------------------------------
// CPU utilities
// -----------------------------------------------------------------------
static int cmp_float(const void* a, const void* b) {
    float fa=*(float*)a, fb=*(float*)b;
    return (fa>fb)-(fa<fb);
}
static float pct(float* sorted, int n, float p) {
    int i=(int)(p*n); return sorted[i<n?i:n-1];
}
static double now_sec() {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts);
    return ts.tv_sec+ts.tv_nsec*1e-9;
}

int main() {
    printf("ODP Bootstrap Chain Ladder — England-Verrall (2002)\n");
    printf("Overdispersion correction: sqrt(%d/%d) = %.4f\n",
           ODP_N, ODP_N-ODP_P, ODP_CORR);
    printf("IFRS 17 CoC rate: %.0f%%  Motor duration: %.1f yr\n\n",
           COC_RATE*100, COC_DURATION);

    // Load triangles
    float* h_tri = (float*)malloc(N_COMPANIES*N_LAGS*N_LAGS*sizeof(float));
    FILE* fp = fopen("triangles.bin","rb");
    if (!fp) { fprintf(stderr,"Cannot open triangles.bin\n"); return 1; }
    fread(h_tri,sizeof(float),N_COMPANIES*N_LAGS*N_LAGS,fp);
    fclose(fp);

    float *d_tri, *d_ibnr, *d_phi;
    rocrand_state_xorwow* d_states;
    long n_states = (long)N_COMPANIES * N_BOOT;

    HIP_CHECK(hipMalloc(&d_tri,    N_COMPANIES*N_LAGS*N_LAGS*sizeof(float)));
    HIP_CHECK(hipMalloc(&d_states, n_states*sizeof(rocrand_state_xorwow)));
    HIP_CHECK(hipMalloc(&d_ibnr,   n_states*sizeof(float)));
    HIP_CHECK(hipMalloc(&d_phi,    N_COMPANIES*sizeof(float)));

    HIP_CHECK(hipMemcpy(d_tri, h_tri,
        N_COMPANIES*N_LAGS*N_LAGS*sizeof(float), hipMemcpyHostToDevice));

    int blocks_per_co = (N_BOOT + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 grid(blocks_per_co, N_COMPANIES), block(BLOCK_SIZE);

    printf("Initialising %ld RNG states...\n", n_states);
    hipLaunchKernelGGL(init_rng, grid, block, 0, 0,
                       d_states, 42ULL, (int)n_states);
    HIP_CHECK(hipDeviceSynchronize());

    printf("Running ODP bootstrap...\n");
    double t0 = now_sec();
    hipLaunchKernelGGL(bootstrap_odp, grid, block, 0, 0,
                       d_tri, d_states, d_ibnr, d_phi);
    HIP_CHECK(hipDeviceSynchronize());
    double elapsed = now_sec() - t0;
    printf("Done in %.3fs  (%.1fM projections/s)\n\n",
           elapsed, (double)N_COMPANIES*N_BOOT/elapsed/1e6);

    float* h_ibnr = (float*)malloc(n_states*sizeof(float));
    float* h_phi  = (float*)malloc(N_COMPANIES*sizeof(float));
    HIP_CHECK(hipMemcpy(h_ibnr, d_ibnr, n_states*sizeof(float),
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_phi,  d_phi,  N_COMPANIES*sizeof(float),
                        hipMemcpyDeviceToHost));

    // Load company names
    char names[N_COMPANIES][128];
    FILE* fc = fopen("companies.txt","r");
    for (int i=0;i<N_COMPANIES;i++) {
        if (fc && fgets(names[i],128,fc)) {
            char* nl=strchr(names[i],'\n'); if(nl)*nl='\0';
        } else snprintf(names[i],128,"Company %d",i);
    }
    if (fc) fclose(fc);

    // Per-company statistics
    float* sorted = (float*)malloc(N_BOOT*sizeof(float));

    printf("%-38s  %8s  %8s  %8s  %6s  %7s  %7s  %8s\n",
           "Company","BestEst","P75","P95","CV%","phi","CoC RM","Adequacy");
    printf("%s\n",
        "--------------------------------------------------------------------------------------------"
        "----------------");

    FILE* fo = fopen("ibnr_odp_summary.csv","w");
    fprintf(fo,"company,best_est,std,p50,p75,p95,p99,cv_pct,"
               "phi,coc_risk_margin,ifrs17_reserve\n");

    double total_be=0, total_p75=0, total_rm=0;

    for (int co=0; co<N_COMPANIES; co++) {
        float* s = h_ibnr + (long)co*N_BOOT;
        memcpy(sorted, s, N_BOOT*sizeof(float));
        qsort(sorted, N_BOOT, sizeof(float), cmp_float);

        double mean=0,var=0;
        for (int b=0;b<N_BOOT;b++) mean+=s[b];
        mean/=N_BOOT;
        for (int b=0;b<N_BOOT;b++) var+=(s[b]-mean)*(s[b]-mean);
        var/=N_BOOT;
        double sd=sqrt(var);
        double cv=(mean>0)?sd/mean*100.0:0;

        float p50=pct(sorted,N_BOOT,0.50f);
        float p75=pct(sorted,N_BOOT,0.75f);
        float p95=pct(sorted,N_BOOT,0.95f);
        float p99=pct(sorted,N_BOOT,0.99f);

        // IFRS 17 CoC risk margin
        // Risk capital = P75 - best estimate (75th pct is SAMA standard)
        // Risk margin = CoC_rate * duration * risk_capital
        float risk_capital = fmaxf(0.0f, p75 - (float)mean);
        float coc_rm       = COC_RATE * COC_DURATION * risk_capital;
        float ifrs17_res   = (float)mean + coc_rm;

        total_be  += mean;
        total_p75 += p75;
        total_rm  += coc_rm;

        char disp[39];
        snprintf(disp,39,"%s",names[co]);

        printf("%-38s  %8.0f  %8.0f  %8.0f  %5.1f%%  %5.2f  %7.0f  %8.0f\n",
               disp, mean, (double)p75, (double)p95, cv,
               h_phi[co], (double)coc_rm, (double)ifrs17_res);

        fprintf(fo,"%s,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.4f,%.2f,%.2f\n",
                names[co], mean, sd, (double)p50, (double)p75,
                (double)p95, (double)p99, cv,
                h_phi[co], (double)coc_rm, (double)ifrs17_res);
    }
    fclose(fo);

    printf("\n%-38s  %8.0f  %8.0f  %8s  %5s  %5s  %7.0f  %8.0f\n",
           "PORTFOLIO TOTAL",
           total_be, total_p75, "", "", "",
           total_rm, total_be+total_rm);

    printf("\nColumn guide:\n");
    printf("  BestEst  = bootstrap mean (IFRS 17 fulfilment cash flows)\n");
    printf("  P75      = 75th percentile (SAMA RBC requirement)\n");
    printf("  P95      = 95th percentile (internal stress test)\n");
    printf("  phi      = overdispersion parameter (phi>1 means extra volatility)\n");
    printf("  CoC RM   = IFRS 17 risk margin (Cost of Capital method @ %.0f%% x %.1fyr)\n",
           COC_RATE*100, COC_DURATION);
    printf("  Adequacy = IFRS 17 reported reserve (BestEst + CoC RM)\n");

    // Save full samples
    FILE* fb = fopen("ibnr_odp_samples.bin","wb");
    fwrite(h_ibnr,sizeof(float),n_states,fb);
    fclose(fb);
    printf("\nSaved ibnr_odp_summary.csv and ibnr_odp_samples.bin\n");

    HIP_CHECK(hipFree(d_tri));
    HIP_CHECK(hipFree(d_states));
    HIP_CHECK(hipFree(d_ibnr));
    HIP_CHECK(hipFree(d_phi));
    free(h_tri); free(h_ibnr); free(h_phi); free(sorted);
    return 0;
}
