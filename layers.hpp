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
		LayerType *layer = new LayerType(args...);

		layer->input = buffers.back();

		Matrix<3, int> *output = new Matrix<3, int>(layer->calc_output_sizes(buffers.back()->get_sizes()));
		buffers.push_back(output);
		layer->output = output;

		layers.push_back(layer);
	}

	void forward() {};

	void print_buffers() {
		for (auto it = buffers.begin(); it != buffers.end(); it++) {
			(*it)->print();
		}
	}
};

class ConvolutionalLayer : public Layer {
  public:
	vector<Matrix<3, int>> kernel;

	array<size_t, 3> kernel_sizes;
	size_t stride;
	size_t filters;

	ConvolutionalLayer(const array<size_t, 3> &kernel_sizes, size_t filters, size_t stride = 1) {
		this->stride = stride;
		this->filters = filters;

		kernel.resize(filters);
		for (int i = 0; i < 3; i++)
			this->kernel_sizes[i] = kernel_sizes[i];

		random_device rd;
		mt19937 gen(rd());
		uniform_int_distribution<int> dist(-5, 5);

		for (size_t f = 0; f < filters; ++f) {
			kernel[f] = Matrix<3, int>(kernel_sizes);

			for (size_t i = 0; i < kernel_sizes[0]; ++i) {
				for (size_t j = 0; j < kernel_sizes[1]; ++j) {
					for (size_t k = 0; k < kernel_sizes[2]; ++k) {
						kernel[f](i, j, k) = dist(gen);
					}
				}
			}
		}
	}

	void forward() override {}

	void backward() override {}

	const array<size_t, 3> calc_output_sizes(const array<size_t, 3> &input_sizes) {
		array<size_t, 3> output_sizes;
		for (int i = 0; i < 3; i++) {
			output_sizes[i] = (input_sizes[i] - kernel_sizes[i]) / stride + 1;
			cout << "LOOK HERE: outpus_sizes[" << i << "] = " << output_sizes[i] << endl;
		}
		return output_sizes;
	}
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
