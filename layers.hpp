#include "matrix.hpp"
#include <array>
#include <cstddef>
#include <random>

class Layer {
  public:
	Matrix<3, int> *input = nullptr;
	Matrix<3, int> *output = nullptr;

	virtual void forward() = 0;
	virtual void backward() = 0;
};

class CNN {
  public:
	vector<Matrix<3, int> *> buffers;
	vector<Layer *> layers;

	CNN(Matrix<3, int> *input_data) { buffers.push_back(input_data); }

	template <typename LayerType, typename... Args> void add_layer(Args... args) {
		Matrix<3, int> *buffer_back = buffers.back();
		LayerType *layer = new LayerType(buffer_back->get_sizes(), args...);

		layer->input = buffer_back;

		Matrix<3, int> *output = new Matrix<3, int>(layer->output_sizes);
		buffers.push_back(output);
		layer->output = output;

		layers.push_back(layer);
	}

	void forward() {}

	void print_buffers() {
		for (auto it = buffers.begin(); it != buffers.end(); it++) {
			(*it)->print();
		}
	}
};

class ConvolutionalLayer : public Layer {
  public:
	vector<Matrix<3, int>> kernel;
	vector<Matrix<2, int>> biases;

	array<size_t, 3> input_sizes;
	array<size_t, 3> output_sizes;
	array<size_t, 3> kernel_sizes;
	array<size_t, 3> biases_sizes;

	size_t stride;
	size_t filters;

	ConvolutionalLayer(const array<size_t, 3> &input_dims, const array<size_t, 2> &kernel_dims, size_t filters,
					   size_t stride = 1) {
		this->input_sizes = input_dims;

		this->stride = stride;
		this->filters = filters;

		kernel.resize(filters);
		biases.resize(filters);

		this->kernel_sizes[0] = input_dims[0];
		this->kernel_sizes[1] = kernel_dims[0];
		this->kernel_sizes[2] = kernel_dims[1];

		this->output_sizes[0] = filters;
		this->output_sizes[1] = (input_sizes[1] - kernel_sizes[1]) / stride + 1;
		this->output_sizes[2] = (input_sizes[2] - kernel_sizes[2]) / stride + 1;

		random_device rd;
		mt19937 gen(rd());
		uniform_int_distribution<int> dist(-5, 5);

		for (size_t f = 0; f < filters; ++f) {
			kernel[f] = Matrix<3, int>(kernel_sizes);
			biases[f] = Matrix<2, int>(kernel_dims);

			// WARNING: note order of indices in the loop nest
			for (size_t i = 0; i < kernel_sizes[1]; ++i) {
				for (size_t j = 0; j < kernel_sizes[2]; ++j) {
					biases[f](i, j) = dist(gen);
					for (size_t k = 0; k < kernel_sizes[0]; ++k) {
						kernel[f](k, j, j) = dist(gen);
					}
				}
			}
		}
	}

	void forward() override {
		for (int i = 0; i < output_sizes[0]; i++) {
			Matrix<2, int> output_sub(output_sizes[1], output_sizes[2]);

			// for (int j = 0; j < kernel_sizes[0]; j++) {
			// 	output_sub = output_sub + (*input)(i).convolute(kernel[i](i));
			// }
			biases[i].print();

			output->assign_slice(output_sub + biases[i], i);
		}
	}

	void backward() override {}
};

class MaxPoolingLayer : public Layer {
  public:
	void forward() {}

	void backward() {}
};

class FullyConnectedLayer : public Layer {
  public:
	void forward() {}

	void backward() {}
};
