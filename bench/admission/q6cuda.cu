// Hardware bound for shared Q6: K separate scan kernels (C1) vs one fused kernel with K accumulators (C2).
// Columns mimic Sirius' GPU layout: date as int32 days, decimals as int64 scaled by 100.
#include <cstdio>
#include <cstdint>
#include <vector>
#include <random>
#include <curand_kernel.h>
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA %s at %d\n", cudaGetErrorString(e), __LINE__); exit(1);} } while (0)

struct P { int lo, hi; long dlo, dhi, q; };
__constant__ P cp[32];

__global__ void gen(int* sd, long* disc, long* qty, long* price, size_t n) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  curandStatePhilox4_32_10_t s; curand_init(42, i, 0, &s);
  for (; i < n; i += (size_t)gridDim.x * blockDim.x) {
    uint4 r = curand4(&s);
    sd[i] = 8036 + r.x % 2526;          // 1992-01-02 .. 1998-12-01 as days since epoch
    disc[i] = r.y % 11;                 // 0.00 .. 0.10
    qty[i] = 100 * (1 + r.z % 50);      // 1 .. 50
    price[i] = 90000 + r.w % 10400000;  // 900.00 .. 104,900.00
  }
}

template <int K>
__global__ void q6(const int* __restrict__ sd, const long* __restrict__ disc, const long* __restrict__ qty,
                   const long* __restrict__ price, size_t n, int base, double* out) {
  double acc[K];
#pragma unroll
  for (int k = 0; k < K; k++) acc[k] = 0;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    int d = sd[i]; long ds = disc[i], qt = qty[i]; double v = (double)(price[i] * ds);
#pragma unroll
    for (int k = 0; k < K; k++) {
      const P& p = cp[base + k];
      bool m = d >= p.lo && d < p.hi && ds >= p.dlo && ds <= p.dhi && qt < p.q;
      acc[k] += m ? v : 0.0;
    }
  }
#pragma unroll
  for (int k = 0; k < K; k++) {
    double v = acc[k];
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
    if ((threadIdx.x & 31) == 0) atomicAdd(&out[base + k], v);
  }
}

template <int K> float run_fused(int* sd, long* di, long* qt, long* pr, size_t n, double* out, int grid) {
  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
  q6<K><<<grid, 256>>>(sd, di, qt, pr, n, 0, out);  // warm
  float best = 1e9;
  for (int r = 0; r < 5; r++) {
    cudaMemset(out, 0, 32 * sizeof(double)); cudaEventRecord(a);
    q6<K><<<grid, 256>>>(sd, di, qt, pr, n, 0, out);
    cudaEventRecord(b); cudaEventSynchronize(b); float ms; cudaEventElapsedTime(&ms, a, b); best = std::min(best, ms);
  }
  return best;
}
float run_separate(int K, int* sd, long* di, long* qt, long* pr, size_t n, double* out, int grid) {
  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b); float best = 1e9;
  for (int r = 0; r < 5; r++) {
    cudaMemset(out, 0, 32 * sizeof(double)); cudaEventRecord(a);
    for (int k = 0; k < K; k++) q6<1><<<grid, 256>>>(sd, di, qt, pr, n, k, out);
    cudaEventRecord(b); cudaEventSynchronize(b); float ms; cudaEventElapsedTime(&ms, a, b); best = std::min(best, ms);
  }
  return best;
}

int main(int argc, char** argv) {
  size_t n = argc > 1 ? atoll(argv[1]) : 600000000ULL;
  int *sd; long *di, *qt, *pr; double* out;
  CK(cudaMalloc(&sd, n * 4)); CK(cudaMalloc(&di, n * 8)); CK(cudaMalloc(&qt, n * 8)); CK(cudaMalloc(&pr, n * 8));
  CK(cudaMalloc(&out, 32 * sizeof(double)));
  gen<<<4096, 256>>>(sd, di, qt, pr, n); CK(cudaDeviceSynchronize());
  int sms; cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0); int grid = sms * 8;
  double bytes = n * 28.0;
  for (const char* dist : {"overlapping", "disjoint"}) {
    std::mt19937 g(11); std::vector<P> ps(32);
    for (int k = 0; k < 32; k++) {
      int y = 1993 + g() % 5; long d = 2 + g() % 8;
      int lo = 8036 + (y - 1992) * 365, hi = lo + 365;
      if (dist[0] == 'd') { lo = 8036 + 2526 * k / 32; hi = 8036 + 2526 * (k + 1) / 32; }
      ps[k] = {lo, hi, d - 1, d + 1, 100L * (24 + (long)(g() % 2))};
    }
    CK(cudaMemcpyToSymbol(cp, ps.data(), sizeof(P) * 32));
    printf("%s (N=%zu rows, %.1f GB per scan)\n", dist, n, bytes / 1e9);
    auto rep = [&](int K, float f) {
      float s = run_separate(K, sd, di, qt, pr, n, out, grid);
      printf("  K=%2d  separate(C1) %8.2f ms  fused(C2) %8.2f ms  gain %5.2fx  fused BW %6.0f GB/s\n", K, s, f, s / f, bytes / (f * 1e6));
    };
    rep(1, run_fused<1>(sd, di, qt, pr, n, out, grid)); rep(2, run_fused<2>(sd, di, qt, pr, n, out, grid));
    rep(4, run_fused<4>(sd, di, qt, pr, n, out, grid)); rep(8, run_fused<8>(sd, di, qt, pr, n, out, grid));
    rep(16, run_fused<16>(sd, di, qt, pr, n, out, grid)); rep(32, run_fused<32>(sd, di, qt, pr, n, out, grid));
  }
  return 0;
}
