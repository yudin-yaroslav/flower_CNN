#include "matrix.hpp"

class ConvolutionalLayer {
  public:
	Matrix<3, int> input;
	Matrix<3, int> kernel;
	Matrix<3, int> output;

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
