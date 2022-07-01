#include <iostream>
#include "cuda_runtime_api.h"
#include "PatternRecognition.h"

#define TILE_WIDTH_S1 8
#define TILE_WIDTH_S2 16
#define TILE_WIDTH_S3 24
#define TILE_WIDTH_S4 32

__global__ void findPatternSharedMemory8(const float* __restrict__ device_target_matrix, const float* __restrict__ device_query_matrix,
		float *device_sads_matrix, const int target_matrix_rows, const int target_matrix_columns, const int query_matrix_rows,
		const int query_matrix_columns, const int sads_matrix_rows, const int sads_matrix_columns) {

	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int col = bx * blockDim.x + tx;
	int row = by * blockDim.y + ty;

	float local_sads_value = 0;

	__shared__ float target_matrix_shared[TILE_WIDTH_S1][TILE_WIDTH_S1];

	for (int phY = 0; phY <= (query_matrix_rows + TILE_WIDTH_S1 - 1) / TILE_WIDTH_S1; ++phY) {
		for (int phX = 0; phX <= (query_matrix_columns + TILE_WIDTH_S1 - 1) / TILE_WIDTH_S1; ++phX) {
			if (phY * TILE_WIDTH_S1 + row < target_matrix_rows && phX * TILE_WIDTH_S1 + col < target_matrix_columns) {
				target_matrix_shared[ty][tx] = device_target_matrix[(phY * TILE_WIDTH_S1 + row) * target_matrix_columns + phX * TILE_WIDTH_S1 + col];
			}

			__syncthreads();

			for (int i = 0; i < TILE_WIDTH_S1; ++i) {
				for (int j = 0; j < TILE_WIDTH_S1; ++j) {

					if (!(phY == 0 && i < ty) && !(phX == 0 && j < tx)
							&& !(tx + (query_matrix_columns % TILE_WIDTH_S1) <= TILE_WIDTH_S1 && query_matrix_columns % TILE_WIDTH_S1 != 0
									&& phX == (((query_matrix_columns + TILE_WIDTH_S1 - 1) / TILE_WIDTH_S1) - 1)
									&& j >= tx + (query_matrix_columns % TILE_WIDTH_S1))
							&& !(tx + (query_matrix_columns % TILE_WIDTH_S1) <= TILE_WIDTH_S1 && query_matrix_columns % TILE_WIDTH_S1 == 0
									&& phX == (((query_matrix_columns + TILE_WIDTH_S1 - 1) / TILE_WIDTH_S1))
									&& j >= tx + (query_matrix_columns % TILE_WIDTH_S1))
							&& !(tx + (query_matrix_columns % TILE_WIDTH_S1) <= TILE_WIDTH_S1 && query_matrix_columns % TILE_WIDTH_S1 != 0
									&& phX > (((query_matrix_columns + TILE_WIDTH_S1 - 1) / TILE_WIDTH_S1) - 1))
							&& !(tx + (query_matrix_columns % TILE_WIDTH_S1) > TILE_WIDTH_S1 && query_matrix_rows % TILE_WIDTH_S1 != 0
									&& phX == ((query_matrix_columns + TILE_WIDTH_S1 - 1) / TILE_WIDTH_S1)
									&& j >= ((tx + (query_matrix_columns % TILE_WIDTH_S1)) % TILE_WIDTH_S1))
							&& !(ty + (query_matrix_rows % TILE_WIDTH_S1) <= TILE_WIDTH_S1 && query_matrix_rows % TILE_WIDTH_S1 != 0
									&& phY == (((query_matrix_rows + TILE_WIDTH_S1 - 1) / TILE_WIDTH_S1) - 1)
									&& i >= ty + (query_matrix_rows % TILE_WIDTH_S1))
							&& !(ty + (query_matrix_rows % TILE_WIDTH_S1) <= TILE_WIDTH_S1 && query_matrix_rows % TILE_WIDTH_S1 == 0
									&& phY == (((query_matrix_rows + TILE_WIDTH_S1 - 1) / TILE_WIDTH_S1))
									&& i >= ty + (query_matrix_rows % TILE_WIDTH_S1))
							&& !(ty + (query_matrix_rows % TILE_WIDTH_S1) <= TILE_WIDTH_S1 && query_matrix_rows % TILE_WIDTH_S1 != 0
									&& phY > (((query_matrix_rows + TILE_WIDTH_S1 - 1) / TILE_WIDTH_S1) - 1))
							&& !(ty + (query_matrix_rows % TILE_WIDTH_S1) > TILE_WIDTH_S1
									&& phY == ((query_matrix_rows + TILE_WIDTH_S1 - 1) / TILE_WIDTH_S1)
									&& i >= ((ty + (query_matrix_rows % TILE_WIDTH_S1)) % TILE_WIDTH_S1))) {
						local_sads_value += std::abs(
								target_matrix_shared[i][j]
										- device_query_matrix[(phY * TILE_WIDTH_S1 + i - ty) * query_matrix_columns + phX * TILE_WIDTH_S1 + j - tx]);
					}
				}
			}
			__syncthreads();
		}
	}
	if (row < sads_matrix_rows && col < sads_matrix_columns) {
		device_sads_matrix[row * sads_matrix_columns + col] = local_sads_value;
	}
}

__global__ void findPatternSharedMemory16(const float* __restrict__ device_target_matrix, const float* __restrict__ device_query_matrix,
		float *device_sads_matrix, const int target_matrix_rows, const int target_matrix_columns, const int query_matrix_rows,
		const int query_matrix_columns, const int sads_matrix_rows, const int sads_matrix_columns) {

	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int col = bx * blockDim.x + tx;
	int row = by * blockDim.y + ty;

	float local_sads_value = 0;

	__shared__ float target_matrix_shared[TILE_WIDTH_S2][TILE_WIDTH_S2];

	for (int phY = 0; phY <= (query_matrix_rows + TILE_WIDTH_S2 - 1) / TILE_WIDTH_S2; ++phY) {
		for (int phX = 0; phX <= (query_matrix_columns + TILE_WIDTH_S2 - 1) / TILE_WIDTH_S2; ++phX) {
			if (phY * TILE_WIDTH_S2 + row < target_matrix_rows && phX * TILE_WIDTH_S2 + col < target_matrix_columns) {
				target_matrix_shared[ty][tx] = device_target_matrix[(phY * TILE_WIDTH_S2 + row) * target_matrix_columns + phX * TILE_WIDTH_S2 + col];
			}

			__syncthreads();

			for (int i = 0; i < TILE_WIDTH_S2; ++i) {
				for (int j = 0; j < TILE_WIDTH_S2; ++j) {

					if (!(phY == 0 && i < ty) && !(phX == 0 && j < tx)
							&& !(tx + (query_matrix_columns % TILE_WIDTH_S2) <= TILE_WIDTH_S2 && query_matrix_columns % TILE_WIDTH_S2 != 0
									&& phX == (((query_matrix_columns + TILE_WIDTH_S2 - 1) / TILE_WIDTH_S2) - 1)
									&& j >= tx + (query_matrix_columns % TILE_WIDTH_S2))
							&& !(tx + (query_matrix_columns % TILE_WIDTH_S2) <= TILE_WIDTH_S2 && query_matrix_columns % TILE_WIDTH_S2 == 0
									&& phX == (((query_matrix_columns + TILE_WIDTH_S2 - 1) / TILE_WIDTH_S2))
									&& j >= tx + (query_matrix_columns % TILE_WIDTH_S2))
							&& !(tx + (query_matrix_columns % TILE_WIDTH_S2) <= TILE_WIDTH_S2 && query_matrix_columns % TILE_WIDTH_S2 != 0
									&& phX > (((query_matrix_columns + TILE_WIDTH_S2 - 1) / TILE_WIDTH_S2) - 1))
							&& !(tx + (query_matrix_columns % TILE_WIDTH_S2) > TILE_WIDTH_S2 && query_matrix_rows % TILE_WIDTH_S2 != 0
									&& phX == ((query_matrix_columns + TILE_WIDTH_S2 - 1) / TILE_WIDTH_S2)
									&& j >= ((tx + (query_matrix_columns % TILE_WIDTH_S2)) % TILE_WIDTH_S2))
							&& !(ty + (query_matrix_rows % TILE_WIDTH_S2) <= TILE_WIDTH_S2 && query_matrix_rows % TILE_WIDTH_S2 != 0
									&& phY == (((query_matrix_rows + TILE_WIDTH_S2 - 1) / TILE_WIDTH_S2) - 1)
									&& i >= ty + (query_matrix_rows % TILE_WIDTH_S2))
							&& !(ty + (query_matrix_rows % TILE_WIDTH_S2) <= TILE_WIDTH_S2 && query_matrix_rows % TILE_WIDTH_S2 == 0
									&& phY == (((query_matrix_rows + TILE_WIDTH_S2 - 1) / TILE_WIDTH_S2))
									&& i >= ty + (query_matrix_rows % TILE_WIDTH_S2))
							&& !(ty + (query_matrix_rows % TILE_WIDTH_S2) <= TILE_WIDTH_S2 && query_matrix_rows % TILE_WIDTH_S2 != 0
									&& phY > (((query_matrix_rows + TILE_WIDTH_S2 - 1) / TILE_WIDTH_S2) - 1))
							&& !(ty + (query_matrix_rows % TILE_WIDTH_S2) > TILE_WIDTH_S2
									&& phY == ((query_matrix_rows + TILE_WIDTH_S2 - 1) / TILE_WIDTH_S2)
									&& i >= ((ty + (query_matrix_rows % TILE_WIDTH_S2)) % TILE_WIDTH_S2))) {
						local_sads_value += std::abs(
								target_matrix_shared[i][j]
										- device_query_matrix[(phY * TILE_WIDTH_S2 + i - ty) * query_matrix_columns + phX * TILE_WIDTH_S2 + j - tx]);
					}
				}
			}
			__syncthreads();
		}
	}
	if (row < sads_matrix_rows && col < sads_matrix_columns) {
		device_sads_matrix[row * sads_matrix_columns + col] = local_sads_value;
	}
}

__global__ void findPatternSharedMemory24(const float* __restrict__ device_target_matrix, const float* __restrict__ device_query_matrix,
		float *device_sads_matrix, const int target_matrix_rows, const int target_matrix_columns, const int query_matrix_rows,
		const int query_matrix_columns, const int sads_matrix_rows, const int sads_matrix_columns) {

	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int col = bx * blockDim.x + tx;
	int row = by * blockDim.y + ty;

	float local_sads_value = 0;

	__shared__ float target_matrix_shared[TILE_WIDTH_S3][TILE_WIDTH_S3];

	for (int phY = 0; phY <= (query_matrix_rows + TILE_WIDTH_S3 - 1) / TILE_WIDTH_S3; ++phY) {
		for (int phX = 0; phX <= (query_matrix_columns + TILE_WIDTH_S3 - 1) / TILE_WIDTH_S3; ++phX) {
			if (phY * TILE_WIDTH_S3 + row < target_matrix_rows && phX * TILE_WIDTH_S3 + col < target_matrix_columns) {
				target_matrix_shared[ty][tx] = device_target_matrix[(phY * TILE_WIDTH_S3 + row) * target_matrix_columns + phX * TILE_WIDTH_S3 + col];
			}

			__syncthreads();

			for (int i = 0; i < TILE_WIDTH_S3; ++i) {
				for (int j = 0; j < TILE_WIDTH_S3; ++j) {

					if (!(phY == 0 && i < ty) && !(phX == 0 && j < tx)
							&& !(tx + (query_matrix_columns % TILE_WIDTH_S3) <= TILE_WIDTH_S3 && query_matrix_columns % TILE_WIDTH_S3 != 0
									&& phX == (((query_matrix_columns + TILE_WIDTH_S3 - 1) / TILE_WIDTH_S3) - 1)
									&& j >= tx + (query_matrix_columns % TILE_WIDTH_S3))
							&& !(tx + (query_matrix_columns % TILE_WIDTH_S3) <= TILE_WIDTH_S3 && query_matrix_columns % TILE_WIDTH_S3 == 0
									&& phX == (((query_matrix_columns + TILE_WIDTH_S3 - 1) / TILE_WIDTH_S3))
									&& j >= tx + (query_matrix_columns % TILE_WIDTH_S3))
							&& !(tx + (query_matrix_columns % TILE_WIDTH_S3) <= TILE_WIDTH_S3 && query_matrix_columns % TILE_WIDTH_S3 != 0
									&& phX > (((query_matrix_columns + TILE_WIDTH_S3 - 1) / TILE_WIDTH_S3) - 1))
							&& !(tx + (query_matrix_columns % TILE_WIDTH_S3) > TILE_WIDTH_S3 && query_matrix_rows % TILE_WIDTH_S3 != 0
									&& phX == ((query_matrix_columns + TILE_WIDTH_S3 - 1) / TILE_WIDTH_S3)
									&& j >= ((tx + (query_matrix_columns % TILE_WIDTH_S3)) % TILE_WIDTH_S3))
							&& !(ty + (query_matrix_rows % TILE_WIDTH_S3) <= TILE_WIDTH_S3 && query_matrix_rows % TILE_WIDTH_S3 != 0
									&& phY == (((query_matrix_rows + TILE_WIDTH_S3 - 1) / TILE_WIDTH_S3) - 1)
									&& i >= ty + (query_matrix_rows % TILE_WIDTH_S3))
							&& !(ty + (query_matrix_rows % TILE_WIDTH_S3) <= TILE_WIDTH_S3 && query_matrix_rows % TILE_WIDTH_S3 == 0
									&& phY == (((query_matrix_rows + TILE_WIDTH_S3 - 1) / TILE_WIDTH_S3))
									&& i >= ty + (query_matrix_rows % TILE_WIDTH_S3))
							&& !(ty + (query_matrix_rows % TILE_WIDTH_S3) <= TILE_WIDTH_S3 && query_matrix_rows % TILE_WIDTH_S3 != 0
									&& phY > (((query_matrix_rows + TILE_WIDTH_S3 - 1) / TILE_WIDTH_S3) - 1))
							&& !(ty + (query_matrix_rows % TILE_WIDTH_S3) > TILE_WIDTH_S3
									&& phY == ((query_matrix_rows + TILE_WIDTH_S3 - 1) / TILE_WIDTH_S3)
									&& i >= ((ty + (query_matrix_rows % TILE_WIDTH_S3)) % TILE_WIDTH_S3))) {
						local_sads_value += std::abs(
								target_matrix_shared[i][j]
										- device_query_matrix[(phY * TILE_WIDTH_S3 + i - ty) * query_matrix_columns + phX * TILE_WIDTH_S3 + j - tx]);
					}
				}
			}
			__syncthreads();
		}
	}
	if (row < sads_matrix_rows && col < sads_matrix_columns) {
		device_sads_matrix[row * sads_matrix_columns + col] = local_sads_value;
	}
}

__global__ void findPatternSharedMemory32(const float* __restrict__ device_target_matrix, const float* __restrict__ device_query_matrix,
		float *device_sads_matrix, const int target_matrix_rows, const int target_matrix_columns, const int query_matrix_rows,
		const int query_matrix_columns, const int sads_matrix_rows, const int sads_matrix_columns) {

	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int col = bx * blockDim.x + tx;
	int row = by * blockDim.y + ty;

	float local_sads_value = 0;

	__shared__ float target_matrix_shared[TILE_WIDTH_S4][TILE_WIDTH_S4];

	for (int phY = 0; phY <= (query_matrix_rows + TILE_WIDTH_S4 - 1) / TILE_WIDTH_S4; ++phY) {
		for (int phX = 0; phX <= (query_matrix_columns + TILE_WIDTH_S4 - 1) / TILE_WIDTH_S4; ++phX) {
			if (phY * TILE_WIDTH_S4 + row < target_matrix_rows && phX * TILE_WIDTH_S4 + col < target_matrix_columns) {
				target_matrix_shared[ty][tx] = device_target_matrix[(phY * TILE_WIDTH_S4 + row) * target_matrix_columns + phX * TILE_WIDTH_S4 + col];
			}

			__syncthreads();

			for (int i = 0; i < TILE_WIDTH_S4; ++i) {
				for (int j = 0; j < TILE_WIDTH_S4; ++j) {

					if (!(phY == 0 && i < ty) && !(phX == 0 && j < tx)
							&& !(tx + (query_matrix_columns % TILE_WIDTH_S4) <= TILE_WIDTH_S4 && query_matrix_columns % TILE_WIDTH_S4 != 0
									&& phX == (((query_matrix_columns + TILE_WIDTH_S4 - 1) / TILE_WIDTH_S4) - 1)
									&& j >= tx + (query_matrix_columns % TILE_WIDTH_S4))
							&& !(tx + (query_matrix_columns % TILE_WIDTH_S4) <= TILE_WIDTH_S4 && query_matrix_columns % TILE_WIDTH_S4 == 0
									&& phX == (((query_matrix_columns + TILE_WIDTH_S4 - 1) / TILE_WIDTH_S4))
									&& j >= tx + (query_matrix_columns % TILE_WIDTH_S4))
							&& !(tx + (query_matrix_columns % TILE_WIDTH_S4) <= TILE_WIDTH_S4 && query_matrix_columns % TILE_WIDTH_S4 != 0
									&& phX > (((query_matrix_columns + TILE_WIDTH_S4 - 1) / TILE_WIDTH_S4) - 1))
							&& !(tx + (query_matrix_columns % TILE_WIDTH_S4) > TILE_WIDTH_S4 && query_matrix_rows % TILE_WIDTH_S4 != 0
									&& phX == ((query_matrix_columns + TILE_WIDTH_S4 - 1) / TILE_WIDTH_S4)
									&& j >= ((tx + (query_matrix_columns % TILE_WIDTH_S4)) % TILE_WIDTH_S4))
							&& !(ty + (query_matrix_rows % TILE_WIDTH_S4) <= TILE_WIDTH_S4 && query_matrix_rows % TILE_WIDTH_S4 != 0
									&& phY == (((query_matrix_rows + TILE_WIDTH_S4 - 1) / TILE_WIDTH_S4) - 1)
									&& i >= ty + (query_matrix_rows % TILE_WIDTH_S4))
							&& !(ty + (query_matrix_rows % TILE_WIDTH_S4) <= TILE_WIDTH_S4 && query_matrix_rows % TILE_WIDTH_S4 == 0
									&& phY == (((query_matrix_rows + TILE_WIDTH_S4 - 1) / TILE_WIDTH_S4))
									&& i >= ty + (query_matrix_rows % TILE_WIDTH_S4))
							&& !(ty + (query_matrix_rows % TILE_WIDTH_S4) <= TILE_WIDTH_S4 && query_matrix_rows % TILE_WIDTH_S4 != 0
									&& phY > (((query_matrix_rows + TILE_WIDTH_S4 - 1) / TILE_WIDTH_S4) - 1))
							&& !(ty + (query_matrix_rows % TILE_WIDTH_S4) > TILE_WIDTH_S4
									&& phY == ((query_matrix_rows + TILE_WIDTH_S4 - 1) / TILE_WIDTH_S4)
									&& i >= ((ty + (query_matrix_rows % TILE_WIDTH_S4)) % TILE_WIDTH_S4))) {
						local_sads_value += std::abs(
								target_matrix_shared[i][j]
										- device_query_matrix[(phY * TILE_WIDTH_S4 + i - ty) * query_matrix_columns + phX * TILE_WIDTH_S4 + j - tx]);
					}
				}
			}
			__syncthreads();
		}
	}
	if (row < sads_matrix_rows && col < sads_matrix_columns) {
		device_sads_matrix[row * sads_matrix_columns + col] = local_sads_value;
	}
}

void findPatternSharedMemory_wrapper(const float* __restrict__ device_target_matrix, const float* __restrict__ device_query_matrix,
		float *device_sads_matrix, const int target_matrix_rows, const int target_matrix_columns, const int query_matrix_rows,
		const int query_matrix_columns, const int sads_matrix_rows, const int sads_matrix_columns, const int TILE_WIDTH) {

	switch (TILE_WIDTH) {
	case 8: {
		dim3 dimGrid(ceil(((float) target_matrix_columns) / TILE_WIDTH_S1), ceil(((float) target_matrix_rows) / TILE_WIDTH_S1));
		dim3 dimBlock(TILE_WIDTH_S1, TILE_WIDTH_S1);

		findPatternSharedMemory8<<<dimGrid, dimBlock>>>(device_target_matrix, device_query_matrix,
				device_sads_matrix, target_matrix_rows, target_matrix_columns, query_matrix_rows, query_matrix_columns,
				sads_matrix_rows, sads_matrix_columns);
		break;
	}

	case 16: {
		dim3 dimGrid(ceil(((float) target_matrix_columns) / TILE_WIDTH_S2), ceil(((float) target_matrix_rows) / TILE_WIDTH_S2));
		dim3 dimBlock(TILE_WIDTH_S2, TILE_WIDTH_S2);

		findPatternSharedMemory16<<<dimGrid, dimBlock>>>(device_target_matrix, device_query_matrix,
				device_sads_matrix, target_matrix_rows, target_matrix_columns, query_matrix_rows, query_matrix_columns,
				sads_matrix_rows, sads_matrix_columns);
		break;
	}

	case 24: {
		dim3 dimGrid(ceil(((float) target_matrix_columns) / TILE_WIDTH_S3), ceil(((float) target_matrix_rows) / TILE_WIDTH_S3));
		dim3 dimBlock(TILE_WIDTH_S3, TILE_WIDTH_S3);

		findPatternSharedMemory24<<<dimGrid, dimBlock>>>(device_target_matrix, device_query_matrix,
				device_sads_matrix, target_matrix_rows, target_matrix_columns, query_matrix_rows, query_matrix_columns,
				sads_matrix_rows, sads_matrix_columns);
		break;
	}

	case 32: {
		dim3 dimGrid(ceil(((float) target_matrix_columns) / TILE_WIDTH_S4), ceil(((float) target_matrix_rows) / TILE_WIDTH_S4));
		dim3 dimBlock(TILE_WIDTH_S4, TILE_WIDTH_S4);

		findPatternSharedMemory32<<<dimGrid, dimBlock>>>(device_target_matrix, device_query_matrix,
				device_sads_matrix, target_matrix_rows, target_matrix_columns, query_matrix_rows, query_matrix_columns,
				sads_matrix_rows, sads_matrix_columns);
		break;
	}
	default:
		std::cerr << "Dimensione tile non prevista";
	}
}

__global__ void findPatternGlobalMemory(const float* __restrict__ device_target_matrix, const float* __restrict__ device_query_matrix,
		float *device_sads_matrix, const int target_matrix_rows, const int target_matrix_columns, const int query_matrix_rows,
		const int query_matrix_columns, const int sads_matrix_rows, const int sads_matrix_columns) {

	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int col = bx * blockDim.x + tx;
	int row = by * blockDim.y + ty;

	float local_sads_value = 0;
	float target_value = 0;
	float query_value = 0;

	if (row < sads_matrix_rows && col < sads_matrix_columns) {
		for (int i = 0; i < query_matrix_rows; i++) {
			for (int j = 0; j < query_matrix_columns; j++) {

				target_value = device_target_matrix[(i + row) * target_matrix_columns + j + col];
				query_value = device_query_matrix[i * query_matrix_columns + j];
				local_sads_value += std::abs(target_value - query_value);
			}
		}
		device_sads_matrix[row * sads_matrix_columns + col] = local_sads_value;
	}
}

void findPatternGlobalMemory_wrapper(const float* __restrict__ device_target_matrix, const float* __restrict__ device_query_matrix,
		float *device_sads_matrix, const int target_matrix_rows, const int target_matrix_columns, const int query_matrix_rows,
		const int query_matrix_columns, const int sads_matrix_rows, const int sads_matrix_columns, const int TILE_WIDTH) {

	dim3 dimGrid(ceil(((float) sads_matrix_columns) / TILE_WIDTH), ceil(((float) sads_matrix_rows) / TILE_WIDTH));
	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);

	findPatternGlobalMemory<<<dimGrid, dimBlock>>>(device_target_matrix, device_query_matrix,
			device_sads_matrix, target_matrix_rows, target_matrix_columns, query_matrix_rows, query_matrix_columns,
			sads_matrix_rows, sads_matrix_columns);
}

__global__ void findPatternGlobalMemoryAliasing(float* device_target_matrix, float* device_query_matrix, float *device_sads_matrix,
		const int target_matrix_rows, const int target_matrix_columns, const int query_matrix_rows, const int query_matrix_columns,
		const int sads_matrix_rows, const int sads_matrix_columns) {

	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int col = bx * blockDim.x + tx;
	int row = by * blockDim.y + ty;

	float local_sads_value = 0;
	float target_value = 0;
	float query_value = 0;

	if (row < sads_matrix_rows && col < sads_matrix_columns) {
		for (int i = 0; i < query_matrix_rows; i++) {
			for (int j = 0; j < query_matrix_columns; j++) {

				target_value = device_target_matrix[(i + row) * target_matrix_columns + j + col];
				query_value = device_query_matrix[i * query_matrix_columns + j];
				local_sads_value += std::abs(target_value - query_value);
			}
		}
		device_sads_matrix[row * sads_matrix_columns + col] = local_sads_value;
	}
}

void findPatternGlobalMemoryAliasing_wrapper(float* device_target_matrix, float* device_query_matrix, float *device_sads_matrix,
		const int target_matrix_rows, const int target_matrix_columns, const int query_matrix_rows, const int query_matrix_columns,
		const int sads_matrix_rows, const int sads_matrix_columns, const int TILE_WIDTH) {

    dim3 dimGrid(ceil(((float) sads_matrix_columns) / TILE_WIDTH), ceil(((float) sads_matrix_rows) / TILE_WIDTH));
	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);

	findPatternGlobalMemoryAliasing<<<dimGrid, dimBlock>>>(device_target_matrix, device_query_matrix,
			device_sads_matrix, target_matrix_rows, target_matrix_columns, query_matrix_rows, query_matrix_columns,
			sads_matrix_rows, sads_matrix_columns);
}

