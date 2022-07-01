#include <iostream>
#include <chrono>
#include <unistd.h>

#include "PatternRecognition.h"
#include "Utils.h"

#include <cuda.h>
#include "cuda_runtime_api.h"

static void CheckCudaErrorAux(const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

/**
 * Check the return value of the CUDA runtime API call and exit
 * the application if the call has failed.
 */
static void CheckCudaErrorAux(const char *file, unsigned line, const char *statement, cudaError_t err) {
	if (err == cudaSuccess)
		return;
	std::cerr << statement << " returned " << cudaGetErrorString(err) << "(" << err << ") at " << file << ":" << line << std::endl;
	exit(1);
}

int main() {
    const int numExecPerTest = 5;

    cudaError_t allocresult;

	int target_matrix_rows = 1500;
	int target_matrix_columns = 1500;

	int query_matrix_rows = 150;
	int query_matrix_columns = 150;

	int sads_matrix_rows = target_matrix_rows - query_matrix_rows + 1;
	int sads_matrix_columns = target_matrix_columns - query_matrix_columns + 1;

	float* target_matrix = generateRandomUniformMatrix(target_matrix_rows * target_matrix_columns, 10);
	float* query_matrix = generateRandomUniformMatrix(query_matrix_rows * query_matrix_columns, 10);

	float* sads_matrix = new float[sads_matrix_rows * sads_matrix_columns];

	float *device_target_matrix;
	float *device_query_matrix;
	float *device_sads_matrix;

	cudaMalloc((void**) &device_target_matrix, target_matrix_rows * target_matrix_columns * sizeof(float));
	cudaMalloc((void**) &device_query_matrix, query_matrix_rows * query_matrix_columns * sizeof(float));
	cudaMalloc((void**) &device_sads_matrix, sads_matrix_rows * sads_matrix_columns * sizeof(float));

	for (int TILE_WIDTH = 8; TILE_WIDTH < 40; TILE_WIDTH += 8) {
		auto t1 = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < numExecPerTest; i++) {

			cudaMemcpy(device_target_matrix, target_matrix, target_matrix_rows * target_matrix_columns * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(device_query_matrix, query_matrix, query_matrix_rows * query_matrix_columns * sizeof(float), cudaMemcpyHostToDevice);

			findPatternGlobalMemory_wrapper(device_target_matrix, device_query_matrix, device_sads_matrix, target_matrix_rows, target_matrix_columns,
					query_matrix_rows, query_matrix_columns, sads_matrix_rows, sads_matrix_columns, TILE_WIDTH);

			allocresult = cudaMemcpy(sads_matrix, device_sads_matrix, sads_matrix_rows * sads_matrix_columns * sizeof(float), cudaMemcpyDeviceToHost);
			CUDA_CHECK_RETURN(allocresult);

        }
		auto t2 = std::chrono::high_resolution_clock::now();
		auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
		std::cout << "global memory with const/__restrict__ with target matrix dim" << target_matrix_rows << " x " << target_matrix_columns
				<< "\t query_matrix dim" << query_matrix_rows << " x " << query_matrix_columns <<"\t TILE_WIDTH " << TILE_WIDTH << "\t time: "
				<< float(duration1 / (numExecPerTest * 1000000.)) << std::endl;

		sleep(5);

	}

    std::cout<< std::endl;

    for (int TILE_WIDTH = 8; TILE_WIDTH < 40; TILE_WIDTH += 8) {
			auto t1 = std::chrono::high_resolution_clock::now();
			for (int i = 0; i < numExecPerTest; i++) {

				cudaMemcpy(device_target_matrix, target_matrix, target_matrix_rows * target_matrix_columns * sizeof(float), cudaMemcpyHostToDevice);
				cudaMemcpy(device_query_matrix, query_matrix, query_matrix_rows * query_matrix_columns * sizeof(float), cudaMemcpyHostToDevice);

				findPatternSharedMemory_wrapper(device_target_matrix, device_query_matrix, device_sads_matrix, target_matrix_rows, target_matrix_columns,
						query_matrix_rows, query_matrix_columns, sads_matrix_rows, sads_matrix_columns, TILE_WIDTH);

				allocresult = cudaMemcpy(sads_matrix, device_sads_matrix, sads_matrix_rows * sads_matrix_columns * sizeof(float), cudaMemcpyDeviceToHost);
				CUDA_CHECK_RETURN(allocresult);
            }
			auto t2 = std::chrono::high_resolution_clock::now();
			auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
			std::cout << "shared memory with target matrix dim" << target_matrix_rows << " x " << target_matrix_columns
					<< "\t query_matrix dim" << query_matrix_rows << " x " << query_matrix_columns <<"\t TILE_WIDTH " << TILE_WIDTH << "\t time: "
					<< float(duration1 / (numExecPerTest * 1000000.)) << std::endl;
			sleep(5);
		}

	std::cout<< std::endl;

	for (int TILE_WIDTH = 8; TILE_WIDTH < 40; TILE_WIDTH += 8) {
				auto t1 = std::chrono::high_resolution_clock::now();
				for (int i = 0; i < numExecPerTest; i++) {

					cudaMemcpy(device_target_matrix, target_matrix, target_matrix_rows * target_matrix_columns * sizeof(float), cudaMemcpyHostToDevice);
					cudaMemcpy(device_query_matrix, query_matrix, query_matrix_rows * query_matrix_columns * sizeof(float), cudaMemcpyHostToDevice);

					findPatternGlobalMemoryAliasing_wrapper(device_target_matrix, device_query_matrix, device_sads_matrix, target_matrix_rows, target_matrix_columns,
							query_matrix_rows, query_matrix_columns, sads_matrix_rows, sads_matrix_columns, TILE_WIDTH);

					allocresult = cudaMemcpy(sads_matrix, device_sads_matrix, sads_matrix_rows * sads_matrix_columns * sizeof(float), cudaMemcpyDeviceToHost);
					CUDA_CHECK_RETURN(allocresult);
				}
				auto t2 = std::chrono::high_resolution_clock::now();
				auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
				std::cout << "global memory wo const/__restrict__ with target matrix dim" << target_matrix_rows << " x " << target_matrix_columns
						<< "\t query_matrix dim" << query_matrix_rows << " x " << query_matrix_columns <<"\t TILE_WIDTH " << TILE_WIDTH << "\t time: "
						<< float(duration1 / (numExecPerTest * 1000000.)) << std::endl;
				sleep(5);
			}

    std::cout<< std::endl;


    cudaFree(device_target_matrix);
	cudaFree(device_query_matrix);
	cudaFree(device_sads_matrix);

	free(target_matrix);
	free(query_matrix);
	free(sads_matrix);


	target_matrix_rows = 2000;
	target_matrix_columns = 2000;

	query_matrix_rows = 200;
	query_matrix_columns = 200;

	sads_matrix_rows = target_matrix_rows - query_matrix_rows + 1;
	sads_matrix_columns = target_matrix_columns - query_matrix_columns + 1;

	target_matrix = generateRandomUniformMatrix(target_matrix_rows * target_matrix_columns, 10);
	query_matrix = generateRandomUniformMatrix(query_matrix_rows * query_matrix_columns, 10);

	sads_matrix = new float[sads_matrix_rows * sads_matrix_columns];


	cudaMalloc((void**) &device_target_matrix, target_matrix_rows * target_matrix_columns * sizeof(float));
	cudaMalloc((void**) &device_query_matrix, query_matrix_rows * query_matrix_columns * sizeof(float));
	cudaMalloc((void**) &device_sads_matrix, sads_matrix_rows * sads_matrix_columns * sizeof(float));

	for (int TILE_WIDTH = 8; TILE_WIDTH < 40; TILE_WIDTH += 8) {
		auto t1 = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < numExecPerTest; i++) {

			cudaMemcpy(device_target_matrix, target_matrix, target_matrix_rows * target_matrix_columns * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(device_query_matrix, query_matrix, query_matrix_rows * query_matrix_columns * sizeof(float), cudaMemcpyHostToDevice);

			findPatternGlobalMemory_wrapper(device_target_matrix, device_query_matrix, device_sads_matrix, target_matrix_rows, target_matrix_columns,
					query_matrix_rows, query_matrix_columns, sads_matrix_rows, sads_matrix_columns, TILE_WIDTH);

			allocresult = cudaMemcpy(sads_matrix, device_sads_matrix, sads_matrix_rows * sads_matrix_columns * sizeof(float), cudaMemcpyDeviceToHost);
			CUDA_CHECK_RETURN(allocresult);
        }
		auto t2 = std::chrono::high_resolution_clock::now();
		auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
		std::cout << "global memory with const/__restrict__ with target matrix dim" << target_matrix_rows << " x " << target_matrix_columns
				<< "\t query_matrix dim" << query_matrix_rows << " x " << query_matrix_columns <<"\t TILE_WIDTH " << TILE_WIDTH << "\t time: "
				<< float(duration1 / (numExecPerTest * 1000000.)) << std::endl;

		sleep(5);

	}
	std::cout<< std::endl;


	for (int TILE_WIDTH = 8; TILE_WIDTH < 40; TILE_WIDTH += 8) {
			auto t1 = std::chrono::high_resolution_clock::now();
			for (int i = 0; i < numExecPerTest; i++) {

				cudaMemcpy(device_target_matrix, target_matrix, target_matrix_rows * target_matrix_columns * sizeof(float), cudaMemcpyHostToDevice);
				cudaMemcpy(device_query_matrix, query_matrix, query_matrix_rows * query_matrix_columns * sizeof(float), cudaMemcpyHostToDevice);

				findPatternSharedMemory_wrapper(device_target_matrix, device_query_matrix, device_sads_matrix, target_matrix_rows, target_matrix_columns,
						query_matrix_rows, query_matrix_columns, sads_matrix_rows, sads_matrix_columns, TILE_WIDTH);

				allocresult = cudaMemcpy(sads_matrix, device_sads_matrix, sads_matrix_rows * sads_matrix_columns * sizeof(float), cudaMemcpyDeviceToHost);
				CUDA_CHECK_RETURN(allocresult);
            }
			auto t2 = std::chrono::high_resolution_clock::now();
			auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
			std::cout << "shared memory with target matrix dim" << target_matrix_rows << " x " << target_matrix_columns
					<< "\t query_matrix dim" << query_matrix_rows << " x " << query_matrix_columns <<"\t TILE_WIDTH " << TILE_WIDTH << "\t time: "
					<< float(duration1 / (numExecPerTest * 1000000.)) << std::endl;
			sleep(5);
		}

	std::cout<< std::endl;

	for (int TILE_WIDTH = 8; TILE_WIDTH < 40; TILE_WIDTH += 8) {
				auto t1 = std::chrono::high_resolution_clock::now();
				for (int i = 0; i < numExecPerTest; i++) {

					cudaMemcpy(device_target_matrix, target_matrix, target_matrix_rows * target_matrix_columns * sizeof(float), cudaMemcpyHostToDevice);
					cudaMemcpy(device_query_matrix, query_matrix, query_matrix_rows * query_matrix_columns * sizeof(float), cudaMemcpyHostToDevice);

					findPatternGlobalMemoryAliasing_wrapper(device_target_matrix, device_query_matrix, device_sads_matrix, target_matrix_rows, target_matrix_columns,
							query_matrix_rows, query_matrix_columns, sads_matrix_rows, sads_matrix_columns, TILE_WIDTH);

					allocresult = cudaMemcpy(sads_matrix, device_sads_matrix, sads_matrix_rows * sads_matrix_columns * sizeof(float), cudaMemcpyDeviceToHost);
					CUDA_CHECK_RETURN(allocresult);
                }
				auto t2 = std::chrono::high_resolution_clock::now();
				auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
				std::cout << "global memory wo const/__restrict__ with target matrix dim" << target_matrix_rows << " x " << target_matrix_columns
						<< "\t query_matrix dim" << query_matrix_rows << " x " << query_matrix_columns <<"\t TILE_WIDTH " << TILE_WIDTH << "\t time: "
						<< float(duration1 / (numExecPerTest * 1000000.)) << std::endl;
				sleep(5);
			}
	std::cout<< std::endl;

	cudaFree(device_target_matrix);
	cudaFree(device_query_matrix);
	cudaFree(device_sads_matrix);

	free(target_matrix);
	free(query_matrix);
	free(sads_matrix);


	target_matrix_rows = 2500;
	target_matrix_columns = 2500;

	query_matrix_rows = 250;
	query_matrix_columns = 250;

	sads_matrix_rows = target_matrix_rows - query_matrix_rows + 1;
	sads_matrix_columns = target_matrix_columns - query_matrix_columns + 1;

	target_matrix = generateRandomUniformMatrix(target_matrix_rows * target_matrix_columns, 10);
	query_matrix = generateRandomUniformMatrix(query_matrix_rows * query_matrix_columns, 10);

	sads_matrix = new float[sads_matrix_rows * sads_matrix_columns];

	cudaMalloc((void**) &device_target_matrix, target_matrix_rows * target_matrix_columns * sizeof(float));
	cudaMalloc((void**) &device_query_matrix, query_matrix_rows * query_matrix_columns * sizeof(float));
	cudaMalloc((void**) &device_sads_matrix, sads_matrix_rows * sads_matrix_columns * sizeof(float));

	for (int TILE_WIDTH = 8; TILE_WIDTH < 40; TILE_WIDTH += 8) {
		auto t1 = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < numExecPerTest; i++) {

			cudaMemcpy(device_target_matrix, target_matrix, target_matrix_rows * target_matrix_columns * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(device_query_matrix, query_matrix, query_matrix_rows * query_matrix_columns * sizeof(float), cudaMemcpyHostToDevice);

			findPatternGlobalMemory_wrapper(device_target_matrix, device_query_matrix, device_sads_matrix, target_matrix_rows, target_matrix_columns,
					query_matrix_rows, query_matrix_columns, sads_matrix_rows, sads_matrix_columns, TILE_WIDTH);

			allocresult = cudaMemcpy(sads_matrix, device_sads_matrix, sads_matrix_rows * sads_matrix_columns * sizeof(float), cudaMemcpyDeviceToHost);
			CUDA_CHECK_RETURN(allocresult);
        }
		auto t2 = std::chrono::high_resolution_clock::now();
		auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
		std::cout << "global memory with const/__restrict__ with target matrix dim" << target_matrix_rows << " x " << target_matrix_columns
				<< "\t query_matrix dim" << query_matrix_rows << " x " << query_matrix_columns <<"\t TILE_WIDTH " << TILE_WIDTH << "\t time: "
				<< float(duration1 / (numExecPerTest * 1000000.)) << std::endl;

		sleep(5);

	}
	std::cout<< std::endl;


	for (int TILE_WIDTH = 8; TILE_WIDTH < 40; TILE_WIDTH += 8) {
			auto t1 = std::chrono::high_resolution_clock::now();
			for (int i = 0; i < numExecPerTest; i++) {

				cudaMemcpy(device_target_matrix, target_matrix, target_matrix_rows * target_matrix_columns * sizeof(float), cudaMemcpyHostToDevice);
				cudaMemcpy(device_query_matrix, query_matrix, query_matrix_rows * query_matrix_columns * sizeof(float), cudaMemcpyHostToDevice);

				findPatternSharedMemory_wrapper(device_target_matrix, device_query_matrix, device_sads_matrix, target_matrix_rows, target_matrix_columns,
						query_matrix_rows, query_matrix_columns, sads_matrix_rows, sads_matrix_columns, TILE_WIDTH);

				allocresult = cudaMemcpy(sads_matrix, device_sads_matrix, sads_matrix_rows * sads_matrix_columns * sizeof(float), cudaMemcpyDeviceToHost);
				CUDA_CHECK_RETURN(allocresult);
            }
			auto t2 = std::chrono::high_resolution_clock::now();
			auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
			std::cout << "shared memory with target matrix dim" << target_matrix_rows << " x " << target_matrix_columns
					<< "\t query_matrix dim" << query_matrix_rows << " x " << query_matrix_columns <<"\t TILE_WIDTH " << TILE_WIDTH << "\t time: "
					<< float(duration1 / (numExecPerTest * 1000000.)) << std::endl;
			sleep(5);
		}

	std::cout<< std::endl;

	for (int TILE_WIDTH = 8; TILE_WIDTH < 40; TILE_WIDTH += 8) {
				auto t1 = std::chrono::high_resolution_clock::now();
				for (int i = 0; i < numExecPerTest; i++) {

					cudaMemcpy(device_target_matrix, target_matrix, target_matrix_rows * target_matrix_columns * sizeof(float), cudaMemcpyHostToDevice);
					cudaMemcpy(device_query_matrix, query_matrix, query_matrix_rows * query_matrix_columns * sizeof(float), cudaMemcpyHostToDevice);

					findPatternGlobalMemoryAliasing_wrapper(device_target_matrix, device_query_matrix, device_sads_matrix, target_matrix_rows, target_matrix_columns,
							query_matrix_rows, query_matrix_columns, sads_matrix_rows, sads_matrix_columns, TILE_WIDTH);

					allocresult = cudaMemcpy(sads_matrix, device_sads_matrix, sads_matrix_rows * sads_matrix_columns * sizeof(float), cudaMemcpyDeviceToHost);
					CUDA_CHECK_RETURN(allocresult);
                }
				auto t2 = std::chrono::high_resolution_clock::now();
				auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
				std::cout << "global memory wo const/__restrict__ with target matrix dim" << target_matrix_rows << " x " << target_matrix_columns
						<< "\t query_matrix dim" << query_matrix_rows << " x " << query_matrix_columns <<"\t TILE_WIDTH " << TILE_WIDTH << "\t time: "
						<< float(duration1 / (numExecPerTest * 1000000.)) << std::endl;
				sleep(5);
			}


	cudaFree(device_target_matrix);
	cudaFree(device_query_matrix);
	cudaFree(device_sads_matrix);

	free(target_matrix);
	free(query_matrix);
	free(sads_matrix);

}
