#include <iostream>
#include "matrix.hpp"
using namespace std;

template <typename T>
Matrix<3, T> convolve4D(const Matrix<3, T> &input,
						const Matrix<4, T> &kernel,
						const Matrix<1, T> &bias)
{
	size_t in_channels = input.sizes()[0];
	size_t in_height = input.sizes()[1];
	size_t in_width = input.sizes()[2];

	size_t out_channels = kernel.sizes()[0];
	size_t k_channels = kernel.sizes()[1];
	size_t k_height = kernel.sizes()[2];
	size_t k_width = kernel.sizes()[3];

	size_t out_height = in_height - k_height + 1;
	size_t out_width = in_width - k_width + 1;

	Matrix<3, T> output({out_channels, out_height, out_width});

	for (size_t oc = 0; oc < out_channels; ++oc)
	{
		for (size_t i = 0; i < out_height; ++i)
		{
			for (size_t j = 0; j < out_width; ++j)
			{
				T sum = T{};
				for (size_t c = 0; c < in_channels; ++c)
				{
					for (size_t ki = 0; ki < k_height; ++ki)
					{
						for (size_t kj = 0; kj < k_width; ++kj)
						{
							sum += input(c, i + ki, j + kj) * kernel(oc, c, ki, kj);
						}
					}
				}
				output(oc, i, j) = sum + bias(oc);
			}
		}
	}

	return output;
}

int main()
{

	return 0;
}