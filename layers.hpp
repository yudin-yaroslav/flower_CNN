#include "matrix.hpp"
#include <array>
#include <cstddef>
#include <random>

class ConvolutionalLayer {
  public:
	vector<Matrix<3, int>> kernel;

	Matrix<3, int> input;
	Matrix<3, int> output;

	ConvolutionalLayer(int number_of_filters, const array<size_t, 3> &kernel_size) {
		kernel.resize(number_of_filters);
		for (int i = 0; i < number_of_filters; ++i)
			kernel[i] = Matrix<3, int>(kernel_size);

		random_device rd;
		mt19937 gen(rd());
		uniform_int_distribution<int> dis(-10, 10);

		for (int f = 0; f < number_of_filters; ++f) {
			for (size_t i = 0; i < kernel_size[0]; ++i) {
				for (size_t j = 0; j < kernel_size[1]; ++j) {
					for (size_t k = 0; k < kernel_size[2]; ++k) {
						kernel[f](i, j, k) = dis(gen);
					}
				}
			}
		}
	}

	void forward() {}

	void backward() {}
};

class MaxPoolingLayer {
  public:
	Matrix<3, int> input;
	Matrix<3, int> kernel;
	Matrix<3, int> output;

	void forward() {}

	void backward() {}
};

class FullyConnectedLayer {
  public:
	Matrix<3, int> input;
	Matrix<3, int> weights;
	Matrix<3, int> output;

	void forward() {}

	void backward() {}
};
