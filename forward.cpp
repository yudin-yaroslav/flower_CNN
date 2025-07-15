#include "convolution.cpp"
#include "matrix.hpp"
#include <iostream>
using namespace std;

template <typename T> Matrix<2, T> maxPooling2D(const Matrix<2, T> &input, size_t pool_size = 2, size_t stride = 2) {
	size_t H = input.get_sizes()[0];
	size_t W = input.get_sizes()[1];

	size_t out_H = (H - pool_size) / stride + 1;
	size_t out_W = (W - pool_size) / stride + 1;

	Matrix<2, T> output({out_H, out_W});

	for (size_t i = 0; i < out_H; ++i) {
		for (size_t j = 0; j < out_W; ++j) {
			T max_val = input(i * stride, j * stride);
			for (size_t m = 0; m < pool_size; ++m) {
				for (size_t n = 0; n < pool_size; ++n) {
					T val = input(i * stride + m, j * stride + n);
					if (val > max_val) {
						max_val = val;
					}
				}
			}
			output(i, j) = max_val;
		}
	}
	return output;
}

Matrix<1, float> forward(const Matrix<2, float> &input, const Matrix<2, float> &kernel, const Matrix<2, float> bias,
						 const Matrix<2, float> &fc_weights) {
	Matrix<2, float> conv_out = bias + convolve2D(input, kernel);
	for (size_t i = 0; i < conv_out.get_sizes()[0]; ++i) {
		for (size_t j = 0; j < conv_out.get_sizes()[1]; ++j) {
			conv_out(i, j) = max(0.0f, conv_out(i, j)); // ReLU
		}
	}
	Matrix<2, float> pooled = maxPooling2D(conv_out, 2, 2); // pooling
	// to flat
	size_t H = pooled.get_sizes()[0], W = pooled.get_sizes()[1];
	Matrix<1, float> flat({H * W});
	for (size_t i = 0; i < H; ++i) {
		for (size_t j = 0; j < W; ++j) {
			flat(i * W + j) = pooled(i, j);
		}
	}
	// попытка сделать fulyy connected (или это не он)
	size_t out_size = fc_weights.get_sizes()[0];
	size_t in_size = fc_weights.get_sizes()[1];

	Matrix<1, float> output({out_size});
	for (size_t i = 0; i < out_size; ++i) {
		float sum = 0;
		for (size_t j = 0; j < in_size; ++j) {
			sum += fc_weights(i, j) * flat(j);
		}
		output(i) = sum;
	}

	return output;
}
