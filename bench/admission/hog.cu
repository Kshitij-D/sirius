// Co-tenant GPU memory hog: wait, grab N GiB, hold, release.
// usage: hog <delay_s> <gib> <hold_s>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <chrono>
#include <cuda_runtime.h>

int main(int argc, char** argv)
{
  if (argc < 4) { std::fprintf(stderr, "usage: hog <delay_s> <gib> <hold_s>\n"); return 2; }
  double delay = std::atof(argv[1]), gib = std::atof(argv[2]), hold = std::atof(argv[3]);
  std::this_thread::sleep_for(std::chrono::duration<double>(delay));
  void* p     = nullptr;
  size_t want = static_cast<size_t>(gib * (1ULL << 30));
  cudaError_t e = cudaMalloc(&p, want);
  std::printf("hog: alloc %.1f GiB -> %s\n", gib, cudaGetErrorString(e));
  std::fflush(stdout);
  if (e == cudaSuccess) {
    cudaMemset(p, 1, want);  // touch it
    cudaDeviceSynchronize();
    std::this_thread::sleep_for(std::chrono::duration<double>(hold));
    cudaFree(p);
    std::printf("hog: released\n");
  }
  return e == cudaSuccess ? 0 : 1;
}
