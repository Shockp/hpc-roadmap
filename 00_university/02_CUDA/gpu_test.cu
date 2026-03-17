#include <cuda_runtime.h>
#include <stdio.h>

__global__ void vectorAdd(float *a, float *b, float *c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    c[i] = a[i] + b[i];
}

int main() {
  int n = 1 << 20; // 1M elements
  size_t size = n * sizeof(float);

  float *h_a = (float *)malloc(size);
  float *h_b = (float *)malloc(size);
  float *h_c = (float *)malloc(size);

  for (int i = 0; i < n; i++) {
    h_a[i] = i;
    h_b[i] = i * 2.0f;
  }

  float *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, size);
  cudaMalloc(&d_b, size);
  cudaMalloc(&d_c, size);

  cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  vectorAdd<<<blocks, threads>>>(d_a, d_b, d_c, n);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("KERNEL ERROR: %s\n", cudaGetErrorString(err));
    return 1;
  }

  cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

  // Verify results
  bool ok = true;
  for (int i = 0; i < n; i++) {
    if (h_c[i] != h_a[i] + h_b[i]) {
      ok = false;
      break;
    }
  }

  printf("GPU compute: %s\n",
         ok ? "PASS - GPU is working correctly" : "FAIL - something is wrong");

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  free(h_a);
  free(h_b);
  free(h_c);
  return 0;
}