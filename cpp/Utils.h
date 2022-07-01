#ifndef PATTERN_RECOGNITION_UTILS_H
#define PATTERN_RECOGNITION_UTILS_H

#include <iostream>
#include <string>
#include <vector>

template<typename T>
int minElementVector(const std::vector<T> vector) {
    int min_el = 0;
    for (int i = 0; i < vector.size(); ++i) {
        if (vector[i] < vector[min_el]) {
            min_el = i;
        }
    }
    return min_el;
}

template<typename T>
void printMatrix(const std::vector<T> &matrix, int rows, int columns, const std::string &name = "") {
    std::cout << std::endl << name << " " << std::endl;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < columns; ++j) {
            std::cout << matrix[columns * i + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

#endif //PATTERN_RECOGNITION_UTILS_H
