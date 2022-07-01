#ifndef PATTERN_RECOGNITION_PATTERNRECOGNITION_H
#define PATTERN_RECOGNITION_PATTERNRECOGNITION_H

#include "Utils.h"
#include <cmath>
#include <iostream>
#include <zconf.h>
#include "Matrix.h"

#ifdef _OPENMP

#include <omp.h>

#endif

template<typename T>
class PatternRecognition {
public:

    PatternRecognition(const Matrix<T> &query_matrix, const Matrix<T> &target_matrix) : query_matrix(
            query_matrix), target_matrix(target_matrix), sads_matrix(
            Matrix<T>((target_matrix.getNumberOfRows() - query_matrix.getNumberOfRows() + 1),
                      (target_matrix.getNumberOfColumns() - query_matrix.getNumberOfColumns() + 1))) {
        cy_coordinate = target_matrix.getNumberOfRows();
        cx_coordinate = target_matrix.getNumberOfColumns();
    }

    void parallelFindPattern(int numThread = -1) {
#ifdef _OPENMP
        if (numThread < 1) {
            numThread = omp_get_num_devices();
        }
#endif
#pragma omp parallel for num_threads(numThread) collapse(2) schedule(static) default(none)
        for (int i = 0; i < sads_matrix.getNumberOfRows(); ++i) {
            for (int j = 0; j < sads_matrix.getNumberOfColumns(); ++j) {

                sads_matrix.setValue(i, j, 0);

                T difference_value = 0;
                for (int k = 0; k < query_matrix.getNumberOfRows(); ++k) {
                    for (int l = 0; l < query_matrix.getNumberOfColumns(); ++l) {

                        T target_value = target_matrix.getValue(i + k, j + l);
                        T query_value = query_matrix.getValue(k, l);
                        difference_value += std::abs(target_value - query_value);

                    }
                }
                sads_matrix.setValue(i, j, difference_value);
            }
        }

        int top_left_coordinate = minElementVector(sads_matrix.getValues());
        cy_coordinate = top_left_coordinate / sads_matrix.getNumberOfColumns();
        cx_coordinate = top_left_coordinate % sads_matrix.getNumberOfColumns();
    }

    void sequentialFindPattern() {
        for (int i = 0; i < sads_matrix.getNumberOfRows(); ++i) {
            for (int j = 0; j < sads_matrix.getNumberOfColumns(); ++j) {

                sads_matrix.setValue(i, j, 0);

                T difference_value = 0;
                for (int k = 0; k < query_matrix.getNumberOfRows(); ++k) {
                    for (int l = 0; l < query_matrix.getNumberOfColumns(); ++l) {

                        T target_value = target_matrix.getValue(i + k, j + l);
                        T query_value = query_matrix.getValue(k, l);
                        difference_value += std::abs(target_value - query_value);

                    }
                }
                sads_matrix.setValue(i, j, difference_value);
            }
        }

        int top_left_coordinate = minElementVector(sads_matrix.getValues());
        cy_coordinate = top_left_coordinate / sads_matrix.getNumberOfColumns();
        cx_coordinate = top_left_coordinate % sads_matrix.getNumberOfColumns();
    }


    void printPatterns() {
        std::cout << std::endl << "cx_coordinate " << cx_coordinate << std::endl;
        std::cout << std::endl << "cy_coordinate " << cy_coordinate << std::endl;

        printMatrix(target_matrix.getValues(), target_matrix.getNumberOfRows(), target_matrix.getNumberOfColumns(),
                    "target");
        printMatrix(query_matrix.getValues(), query_matrix.getNumberOfRows(), query_matrix.getNumberOfColumns(),
                    "query");
        printMatrix(sads_matrix.getValues(), sads_matrix.getNumberOfRows(), sads_matrix.getNumberOfColumns(), "sads");
    }

    const Matrix<T> &getQueryMatrix() const {
        return query_matrix;
    }

    const Matrix<T> &getTargetMatrix() const {
        return target_matrix;
    }

    const Matrix<T> &getSadsMatrix() const {
        return sads_matrix;
    }


    int getCxCoordinate() const {
        return cx_coordinate;
    }

    int getCyCoordinate() const {
        return cy_coordinate;
    }

private:
    const Matrix<T> &query_matrix;
    const Matrix<T> &target_matrix;
    Matrix<T> sads_matrix;
    int cx_coordinate;
    int cy_coordinate;
};

#endif //PATTERN_RECOGNITION_PATTERNRECOGNITION_H
