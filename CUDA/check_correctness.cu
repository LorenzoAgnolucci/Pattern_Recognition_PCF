#include <iostream>
#include <thread>

#include "PatternRecognition.h"
#include "Utils.h"

#include <cuda.h>
#include "cuda_runtime_api.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>


static void CheckCudaErrorAux(const char *, unsigned, const char *, cudaError_t);

#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

/**
 * Check the return value of the CUDA runtime API call and exit
 * the application if the call has failed.
 */
static void CheckCudaErrorAux(const char *file, unsigned line, const char *statement, cudaError_t err) {
    if (err == cudaSuccess)
        return;
    std::cerr << statement << " returned " << cudaGetErrorString(err) << "(" << err << ") at " << file << ":" << line
              << std::endl;
    exit(1);
}

int main() {
    std::string target_image_path = "../images/input.jpg";
    std::string query_image_path = "../images/cropped_input.jpg";

    cv::Mat target_image_color = cv::imread(target_image_path, cv::IMREAD_COLOR);
    cv::Mat target_image;
    cv::cvtColor(target_image_color, target_image, cv::COLOR_BGR2GRAY);
    cv::Mat query_image = cv::imread(query_image_path, cv::IMREAD_GRAYSCALE);

    cudaError_t allocresult;

    int target_matrix_rows = target_image.rows;
    int target_matrix_columns = target_image.cols;

    int query_matrix_rows = query_image.rows;
    int query_matrix_columns = query_image.cols;

    int sads_matrix_rows = target_matrix_rows - query_matrix_rows + 1;
    int sads_matrix_columns = target_matrix_columns - query_matrix_columns + 1;

    float *target_matrix = cvMat2Array(target_image);
    float *query_matrix = cvMat2Array(query_image);

    auto *sads_matrix = new float[sads_matrix_rows * sads_matrix_columns];

    float *device_target_matrix;
    float *device_query_matrix;
    float *device_sads_matrix;

    cudaMalloc((void **) &device_target_matrix, target_matrix_rows * target_matrix_columns * sizeof(float));
    cudaMalloc((void **) &device_query_matrix, query_matrix_rows * query_matrix_columns * sizeof(float));
    cudaMalloc((void **) &device_sads_matrix, sads_matrix_rows * sads_matrix_columns * sizeof(float));

    cudaMemcpy(device_target_matrix, target_matrix, target_matrix_rows * target_matrix_columns * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(device_query_matrix, query_matrix, query_matrix_rows * query_matrix_columns * sizeof(float),
               cudaMemcpyHostToDevice);

    int TILE_WIDTH = 16;
    findPatternGlobalMemory_wrapper(device_target_matrix, device_query_matrix, device_sads_matrix, target_matrix_rows,
                                    target_matrix_columns,
                                    query_matrix_rows, query_matrix_columns, sads_matrix_rows, sads_matrix_columns,
                                    TILE_WIDTH);

    allocresult = cudaMemcpy(sads_matrix, device_sads_matrix, sads_matrix_rows * sads_matrix_columns * sizeof(float),
                             cudaMemcpyDeviceToHost);
    CUDA_CHECK_RETURN(allocresult);

    bool equal = check_correctness(target_matrix, query_matrix, sads_matrix, target_matrix_rows, target_matrix_columns,
                      query_matrix_rows, query_matrix_columns, sads_matrix_rows, sads_matrix_columns);
    if (equal){
        std::cout<< "Le implementazioni sequenziale e parallela hanno il medesimo risultato!" << std::endl;
    }
    else{
        std::cout<< "Le implementazioni sequenziale e parallela NON hanno il medesimo risultato" << std::endl;
    }

    int top_left_coordinate = minElementMatrix(sads_matrix, sads_matrix_rows, sads_matrix_columns);

    int cy_coordinate = top_left_coordinate / sads_matrix_columns;
    int cx_coordinate = top_left_coordinate % sads_matrix_columns;

    auto points = std::make_pair<cv::Point, cv::Point>(
            cv::Point(cx_coordinate, cy_coordinate),
            cv::Point(cx_coordinate + query_matrix_columns,
                      cy_coordinate + query_matrix_rows));

    cv::rectangle(target_image_color, std::get<0>(points), std::get<1>(points),
                  cv::Scalar(0, 255, 0), 2);

    cv::imwrite("../images/pattern_output_cuda.jpg", target_image_color);

    cudaFree(device_target_matrix);
    cudaFree(device_query_matrix);
    cudaFree(device_sads_matrix);

    free(target_matrix);
    free(query_matrix);
    free(sads_matrix);


}
