// Co-tenant GPU memory hog: wait, grab N GiB (retrying until it fits or a timeout), hold, release.
// usage: hog <delay_s> <gib> <hold_s> [timeout_s]
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <cuda_runtime.h>

int main(int argc, char** argv)
{
  if (argc < 4) { std::fprintf(stderr, "usage: hog <delay_s> <gib> <hold_s> [timeout_s]\n"); return 2; }
  double delay = std::atof(argv[1]), gib = std::atof(argv[2]), hold = std::atof(argv[3]);
  double timeout = argc > 4 ? std::atof(argv[4]) : 30.0;
  std::this_thread::sleep_for(std::chrono::duration<double>(delay));
  size_t want = static_cast<size_t>(gib * (1ULL << 30));
  void* p     = nullptr;
  auto t0     = std::chrono::steady_clock::now();
  cudaError_t e;
  int tries = 0;
  while (true) {
    e = cudaMalloc(&p, want);
    ++tries;
    double waited = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (e == cudaSuccess || waited > timeout) {
      std::printf("hog: alloc %.1f GiB -> %s after %.2f s (%d tries)\n", gib, cudaGetErrorString(e), waited, tries);
      std::fflush(stdout);
      break;
    }
    cudaGetLastError();  // clear the error before retrying
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  if (e == cudaSuccess) {
    cudaMemset(p, 1, want);
    cudaDeviceSynchronize();
    std::this_thread::sleep_for(std::chrono::duration<double>(hold));
    cudaFree(p);
    std::printf("hog: released\n");
  }
  return e == cudaSuccess ? 0 : 1;
}
