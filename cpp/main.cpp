#include <iostream>
#include "Matrix.h"
#include "PatternRecognition.h"
#include <chrono>


int main() {
    const int numExecPerTest = 5;
    const int maxNumThreads = 11;

    //Test1
    int trow = 1500;
    int tcol = 1500;
    int qrow = 150;
    int qcol = 150;


    for (int numThread = 1; numThread < maxNumThreads; ++numThread) {
        auto t1 = std::chrono::high_resolution_clock::now();
        Matrix<int> target(trow, tcol);
        Matrix<int> query(qrow, qcol);

        target.generateRandomUniformMatrix(10);
        query.generateRandomUniformMatrix(10);
        for (int i = 0; i < numExecPerTest; ++i) {
            PatternRecognition<int> pattern(query, target);
            if (numThread == 1) {
                pattern.sequentialFindPattern();
            } else {
                pattern.parallelFindPattern(numThread);
            }
        }

        auto t2 = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

        std::cout << std::endl << "Targetdim " << trow << "x" << tcol << "\tquerydim " << qrow << "x" << qcol
                  << "\twith numThread " << numThread << "  time:" << float(duration / (numExecPerTest * 1000000.)) << std::endl;
        sleep(5);
    }
    std::cout << std::endl << std::endl;

    //-------------------------------------------------------------------------------------------------------------
    //Test2

    trow = 2000;
    tcol = 2000;
    qrow = 200;
    qcol = 200;
    for (int numThread = 1; numThread < maxNumThreads; ++numThread) {
        auto t1 = std::chrono::high_resolution_clock::now();
        Matrix<int> target(trow, tcol);
        Matrix<int> query(qrow, qcol);

        target.generateRandomUniformMatrix(10);
        query.generateRandomUniformMatrix(10);
        for (int i = 0; i < numExecPerTest; ++i) {
            PatternRecognition<int> pattern(query, target);
            if (numThread == 1) {
                pattern.sequentialFindPattern();
            } else {
                pattern.parallelFindPattern(numThread);
            }
        }

        auto t2 = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

        std::cout << std::endl << "Targetdim " << trow << "x" << tcol << "\tquerydim " << qrow << "x" << qcol
                  << "\twith numThread " << numThread << "  time:" << float(duration / (numExecPerTest * 1000000.)) << std::endl;
        sleep(5);
    }

    std::cout << std::endl << std::endl;

    //------------------------------------------------------------------------------------------------------------
    //Test3

    trow = 2500;
    tcol = 2500;
    qrow = 250;
    qcol = 250;
    for (int numThread = 1; numThread < maxNumThreads; ++numThread) {
        auto t1 = std::chrono::high_resolution_clock::now();
        Matrix<int> target(trow, tcol);
        Matrix<int> query(qrow, qcol);

        target.generateRandomUniformMatrix(10);
        query.generateRandomUniformMatrix(10);
        for (int i = 0; i < numExecPerTest; ++i) {
            PatternRecognition<int> pattern(query, target);
            if (numThread == 1) {
                pattern.sequentialFindPattern();
            } else {
                pattern.parallelFindPattern(numThread);
            }
        }

        auto t2 = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

        std::cout << std::endl << "Targetdim " << trow << "x" << tcol << "\tquerydim " << qrow << "x" << qcol
                  << "\twith numThread " << numThread << "  time:" << float(duration / (numExecPerTest * 1000000.)) << std::endl;
        sleep(5);
    }
}
