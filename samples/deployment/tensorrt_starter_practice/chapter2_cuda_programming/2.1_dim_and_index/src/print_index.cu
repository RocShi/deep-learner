#include <cuda_runtime.h>
#include <iostream>

__global__ void PrintIdxKernel() {
  printf("  PrintIndexKernel - block idx: (%3d, %3d, %3d), thread idx: (%3d, "
         "%3d, %3d)\n",
         blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y,
         threadIdx.z);
}

__global__ void PrintDimKernel() {
  printf("  PrintDimKernel - gridDim: (%3d, %3d, %3d), blockDim: (%3d, "
         "%3d, %3d)\n",
         gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z);
}

__global__ void PrintThreadIdxPerBlockKernel() {
  const int thread_idx = threadIdx.z * blockDim.x * blockDim.y +
                         threadIdx.y * blockDim.x + threadIdx.x;
  printf("  PrintThreadIdxPerBlockKernel - block idx: (%3d, %3d, %3d), thread "
         "idx: (%3d)\n",
         blockIdx.x, blockIdx.y, blockIdx.z, thread_idx);
}

__global__ void PrintThreadIdxPerGridKernel() {
  const int thread_per_block = blockDim.x * blockDim.y * blockDim.z;
  const int block_idx =
      blockIdx.z * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x;
  const int thread_idx = block_idx * thread_per_block +
                         threadIdx.z * blockDim.x * blockDim.y +
                         threadIdx.y * blockDim.x + threadIdx.x;
  printf("  PrintThreadIdxPerGridKernel - thread per block: (%3d), block idx: "
         "(%3d), "
         "thread idx: (%3d)\n",
         thread_per_block, block_idx, thread_idx);
}

__global__ void PrintThreadCoordInGridKernel() {
  const int thread_coord_x = blockIdx.x * blockDim.x + threadIdx.x;
  const int thread_coord_y = blockIdx.y * blockDim.y + threadIdx.y;
  const int thread_coord_z = blockIdx.z * blockDim.z + threadIdx.z;
  printf("  PrintThreadCoordInGridKernel - thread coord: (%3d, %3d, %3d)\n",
         thread_coord_x, thread_coord_y, thread_coord_z);
}

void Process1DInput(const int input_size) {
  std::cout << "Process1DInput: input_size = " << input_size << std::endl;

  const int thread_per_block = 4;
  const int block_num = input_size / thread_per_block;
  const dim3 grid_dim(block_num, 1, 1);
  const dim3 block_dim(thread_per_block, 1, 1);

  PrintIdxKernel<<<grid_dim, block_dim>>>();
  PrintDimKernel<<<grid_dim, block_dim>>>();
  PrintThreadIdxPerBlockKernel<<<grid_dim, block_dim>>>();
  PrintThreadIdxPerGridKernel<<<grid_dim, block_dim>>>();
  PrintThreadCoordInGridKernel<<<grid_dim, block_dim>>>();

  cudaDeviceSynchronize();
}

void Process2DInput(const int input_width, const int input_height) {
  std::cout << "Process2DInput: input_width = " << input_width
            << ", input_height = " << input_height << std::endl;

  const int thread_per_block_x = 2;
  const int thread_per_block_y = 3;
  const int block_num_x = input_width / thread_per_block_x;
  const int block_num_y = input_height / thread_per_block_y;
  const dim3 grid_dim(block_num_x, block_num_y, 1);
  const dim3 block_dim(thread_per_block_x, thread_per_block_y, 1);

  PrintIdxKernel<<<grid_dim, block_dim>>>();
  PrintDimKernel<<<grid_dim, block_dim>>>();
  PrintThreadIdxPerBlockKernel<<<grid_dim, block_dim>>>();
  PrintThreadIdxPerGridKernel<<<grid_dim, block_dim>>>();
  PrintThreadCoordInGridKernel<<<grid_dim, block_dim>>>();

  cudaDeviceSynchronize();
}

int main() {
  std::cout << "\nRunning 2.1_dim_and_index..." << std::endl;

  std::cout << "--------------------------------" << std::endl;
  Process1DInput(12);
  std::cout << "--------------------------------" << std::endl;
  Process2DInput(4, 3);
  std::cout << "--------------------------------" << std::endl;

  return 0;
}
