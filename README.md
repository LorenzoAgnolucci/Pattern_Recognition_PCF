# Pattern Recognition PCF
Pattern Recognition parallel implementation for the Parallel Computing Fundamentals PhD course.


## Table of Contents

* [About the Project](#about-the-project)
  * [Built With](#built-with)
* [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
* [Usage](#usage)
* [Authors](#authors)
* [Acknowledgments](#acknowledgments)

## About The Project
We developed the 2D Pattern Recognition algorithm in two different implementations:
the first one written in C++ using OpenMP, the second one based on GPU written with CUDA.  
* The CUDA implementation is under the ```CUDA``` directory
* The C++ (with OpenMP) implementation is under the ```cpp``` directory

For more details about algorithm and performance comparison see the [report](https://github.com/LorenzoAgnolucci/Pattern_Recognition_PCF/blob/master/Report/report.pdf).


### Built With

* [OpenCV](https://opencv.org/)
* [OpenMP](https://www.openmp.org/)
* [CUDA](https://developer.nvidia.com/cuda-zone)

## Getting Started
To get a local copy up and running follow these simple steps on Ubuntu.

### Prerequisites
* For running the CUDA implementation it is necessary to have a NVIDIA CUDA capable GPU.
For installing the CUDA toolkit you can follow the [official site](https://developer.nvidia.com/cuda-toolkit) or simply run
```sh
sudo apt install nvidia-cuda-toolkit
```

* OpenCV
```sh
sudo apt install libopencv-dev
```

### Installation
1. Clone the repo
```sh
git clone https://github.com/LorenzoAgnolucci/Pattern_Recognition_PCF.git
```
2. Compile the code
```sh
cd Pattern_Recognition_PCF/
mkdir build
cmake -B build -DCMAKE_BUILD_TYPE=Release
cd build
make
```

## Usage
In the ```build``` directory you will find 4 different executables:

1. Run ```./Pattern_Recognition_Cpp_correctness``` to assess the correctness of OpenMP parallel implementation compared to the sequential one. Furthermore, the executable applies the algorithm to the target image ```images/input.jpg``` and the query image ```images/cropped_input.jpg``` and generates the output in ```images/pattern_output_cpp.jpg```.

2. Run ```./Pattern_Recognition_Cpp_benchmark``` to reproduce the OpenMP experimental results presented in the report.

3. Run ```./Pattern_Recognition_CUDA_correctness``` to assess the correctness of CUDA parallel implementation compared to the sequential one. Furthermore, the executable applies the algorithm to the target image ```images/input.jpg``` and the query image ```images/cropped_input.jpg``` and generates the output in ```images/pattern_output_cuda.jpg```.

4. Run ```./Pattern_Recognition_CUDA_benchmark``` to reproduce the CUDA experimental results presented in the report.

## Authors
* **Lorenzo Agnolucci**
* **Alberto Baldrati**

## Acknowledgments
Parallel Computing Fundamentals © Course held by Professor [Roberto Giorgi](https://scholar.google.co.uk/citations?user=BhrfWxAAAAAJ&) and Professor [Marco Procaccini](https://scholar.google.co.uk/citations?user=LlNwv_gAAAAJ&hl=it&oi=ao) - Smart Computing Ph.D. course @[University of Siena](https://www.unisi.it/)
