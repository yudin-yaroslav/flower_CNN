#include "matrix.hpp"
#include <iostream>
using namespace std;

template <typename T> Matrix<2, T> convolve2D(const Matrix<2, T> &input, const Matrix<2, T> &kernel) {
	size_t input_width = input.get_sizes()[0];
	size_t input_height = input.get_sizes()[1];

	size_t kernel_width = kernel.get_sizes()[0];
	size_t kernel_height = kernel.get_sizes()[1];

	size_t output_width = input_width - kernel_width + 1;
	size_t output_height = input_height - kernel_height + 1;

	Matrix<2, T> result({output_width, output_height});
	Matrix<2, T> sub_input;

	for (size_t i = 0; i < output_width; i++) {
		for (size_t j = 0; j < output_height; j++) {
			sub_input = input.slice({i, i + kernel_width, j, j + kernel_height});

			result(i, j) = sub_input * kernel;
		}
	}

	return result;
}

int main() {
	Matrix<2, int> input({3, 3});
	int input_data[9] = {1, 6, 2, 5, 3, 1, 7, 0, 4};

	for (size_t i = 0; i < 3; i++) {
		for (size_t j = 0; j < 3; j++) {
			input(i, j) = input_data[3 * i + j];
		}
	}

	Matrix<2, int> kernel({2, 2});
	int kernel_data[4] = {1, 2, -1, 0};

	for (size_t i = 0; i < 2; i++) {
		for (size_t j = 0; j < 2; j++) {
			kernel(i, j) = kernel_data[2 * i + j];
		}
	}

	Matrix<2, int> result = convolve2D(input, kernel);

	result.print();

	return 0;
}
