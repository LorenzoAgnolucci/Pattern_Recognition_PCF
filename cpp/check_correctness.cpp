#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include "Matrix.h"
#include "PatternRecognition.h"

using namespace cv;

int main() {
    std::string target_image_path = "../images/input.jpg";
    std::string query_image_path = "../images/cropped_input.jpg";

    cv::Mat target_image_color = cv::imread(target_image_path, cv::IMREAD_COLOR);
    cv::Mat target_image;
    cv::cvtColor(target_image_color, target_image, cv::COLOR_BGR2GRAY);
    cv::Mat query_image = cv::imread(query_image_path, cv::IMREAD_GRAYSCALE);

    auto target_image_matrix = Matrix<int>(target_image);
    auto query_image_matrix = Matrix<int>(query_image);

    PatternRecognition<int> pattern_sequential(query_image_matrix, target_image_matrix);
    pattern_sequential.sequentialFindPattern();
    PatternRecognition<int> pattern_parallel(query_image_matrix, target_image_matrix);
    pattern_parallel.parallelFindPattern(-1);

    bool are_equal = true;
    for(int i =0; i<pattern_sequential.getSadsMatrix().getNumberOfColumns() * pattern_sequential.getSadsMatrix().getNumberOfRows(); i++){
        if (pattern_sequential.getSadsMatrix().getValues()[i] != pattern_parallel.getSadsMatrix().getValues()[i]) {
           are_equal = false;
           break;
        }
    }
    if (are_equal){
        std::cout<< "Le implementazioni sequenziale e parallela hanno il medesimo risultato!" << std::endl;
    }
    else{
        std::cout<< "Le implementazioni sequenziale e parallela NON hanno il medesimo risultato" << std::endl;
    }

    auto points = std::make_pair<cv::Point, cv::Point>(
            cv::Point(pattern_sequential.getCxCoordinate(), pattern_sequential.getCyCoordinate()),
            cv::Point(pattern_sequential.getCxCoordinate() + query_image_matrix.getNumberOfColumns(),
                      pattern_sequential.getCyCoordinate() + query_image_matrix.getNumberOfRows()));

    cv::rectangle(target_image_color, std::get<0>(points), std::get<1>(points),
                  cv::Scalar(0, 255, 0), 2);

    cv::imwrite("../images/pattern_output_cpp.jpg", target_image_color);


    return 0;
}