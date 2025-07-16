#include <cstddef>
#include <iostream>
#include <random>
#include <vector>

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

// using namespace std;
using namespace Eigen;
using std::cout, std::endl;

class Layer {
  public:
	Tensor<float, 3> *input = nullptr;
	Tensor<float, 3> *output = nullptr;

	virtual void forward() = 0;
	virtual void backward() = 0;

	virtual ~Layer() = default;
};

class ConvolutionalLayer : public Layer {
  public:
	std::vector<Tensor<float, 2>> kernel;
	std::vector<Tensor<float, 2>> biases;

	array<Index, 3> input_sizes;
	array<Index, 3> output_sizes;
	array<Index, 3> kernel_sizes;
	array<Index, 3> biases_sizes;

	size_t stride;
	size_t filters;

	ConvolutionalLayer(const array<Index, 3> &input_dims, const array<Index, 2> &kernel_dims, Index filters, Index stride) {
		this->input_sizes = input_dims;

		this->stride = stride;
		this->filters = filters;

		kernel.resize(filters);
		biases.resize(filters);

		this->kernel_sizes = {input_dims[0], kernel_dims[0], kernel_dims[1]};

		this->output_sizes = {filters, (input_sizes[1] - kernel_sizes[1]) / stride + 1,
							  (input_sizes[2] - kernel_sizes[2]) / stride + 1};

		for (size_t f = 0; f < filters; ++f) {
			kernel[f] = Tensor<float, 2>(kernel_sizes[1], kernel_sizes[2]);
			biases[f] = Tensor<float, 2>(output_sizes[1], output_sizes[2]);

			kernel[f].setRandom();
			biases[f].setRandom();
		}
	}

	void forward() override {
		for (Index f = 0; f < filters; f++) {
			Tensor<float, 2> output_sub(output_sizes[1], output_sizes[2]);
			output_sub.setZero();

			for (Index c = 0; c < kernel_sizes[0]; c++) {
				Tensor<float, 2> cross_correlated(output_sizes[1], output_sizes[2]);

				for (Index x = 0; x < output_sizes[1]; x++) {
					for (Index y = 0; y < output_sizes[2]; y++) {
						Index in_x = x * stride;
						Index in_y = y * stride;

						float sum = 0.0;
						for (Index u = 0; u < kernel_sizes[1]; u++) {
							for (Index v = 0; v < kernel_sizes[2]; v++) {
								sum += (*input)(c, in_x + u, in_y + v) * kernel[f](u, v);
							}
						}

						cross_correlated(x, y) = sum;
					}
				}

				output_sub += cross_correlated;
			}

			output->chip(f, 0) = output_sub + biases[f];
		}
	}

	void backward() override {}
};

// class MaxPoolingLayer : public Layer {
//   public:
// 	array<size_t, 3> input_sizes;
// 	array<size_t, 2> kernel_sizes;
// 	array<size_t, 3> output_sizes;
//
// 	size_t stride;
//
// 	MaxPoolingLayer(const array<size_t, 3> &input_dims, const array<size_t, 2> &kernel_dims, size_t stride) {
// 		this->input_sizes = input_dims;
// 		this->kernel_sizes = kernel_dims;
// 		this->stride = stride;
//
// 		this->output_sizes = {input_sizes[0], (input_sizes[1] - kernel_sizes[0]) / stride + 1,
// 							  (input_sizes[2] - kernel_sizes[1]) / stride + 1};
// 	}
//
// 	void forward() {
// 		for (size_t c = 0; c < output_sizes[0]; ++c) {
// 			for (size_t i = 0; i < output_sizes[1]; ++i) {
// 				for (size_t j = 0; j < output_sizes[2]; ++j) {
//
// 					float max_val = -INFINITY;
// 					for (size_t x = 0; x < kernel_sizes[0]; ++x) {
// 						for (size_t y = 0; y < kernel_sizes[1]; ++y) {
// 							size_t in_i = i * stride + x;
// 							size_t in_j = j * stride + y;
//
// 							max_val = max(max_val, (*input)(c, in_i, in_j));
// 						}
// 					}
// 					(*output)(c, i, j) = max_val;
// 				}
// 			}
// 		}
// 	}
//
// 	void backward() {}
// };
//
// class FullyConnectedLayer : public Layer {
//   public:
// 	Matrix<2, float> weights;
//
// 	size_t input_size;
// 	size_t output_size;
//
// 	FullyConnectedLayer(size_t input_dim, size_t output_dim) {
// 		weights = Matrix<2, float>(output_dim, input_dim);
// 		this->input_size = input_dim;
// 		this->output_size = output_dim;
//
// 		random_device rd;
// 		mt19937 gen(rd());
// 		uniform_real_distribution<float> dist(-5, 5);
//
// 		for (size_t i = 0; i < output_dim; i++) {
// 			for (size_t j = 0; j < input_dim; j++) {
//
// 				weights(i, j) = dist(gen);
// 			}
// 		}
// 	}
//
// 	void forward() override {
// 		Matrix<2, float> output_2d = weights.mat_multiply((*input)(0));
// 		output->data().assign(output_2d.data().begin(), output_2d.data().end());
// 	}
//
// 	void backward() override {}
// };
//
// class ReLULayer : public Layer {
//   public:
// 	ReLULayer() {};
//
// 	void forward() override {
// 		vector<float> &in = input->data();
// 		vector<float> &out = output->data();
//
// 		for (size_t i = 0; i < in.size(); i++) {
// 			out[i] = max(0.0f, in[i]);
// 		}
// 	}
//
// 	void backward() override {}
// };
//
// class SoftMaxLayer : public Layer {
//   public:
// 	array<size_t, 3> input_sizes;
//
// 	SoftMaxLayer(array<size_t, 3> input_dims) { this->input_sizes = input_dims; };
//
// 	void forward() override {
// 		vector<float> &in = input->data();
// 		vector<float> &out = output->data();
//
// 		float max_val = *max_element(in.begin(), in.end());
// 		cout << max_val << endl;
//
// 		float sum = 0.0f;
// 		for (size_t i = 0; i < in.size(); ++i) {
// 			if (in[i] != 0) {
// 				sum += exp(in[i]);
// 			}
// 		}
//
// 		for (size_t i = 0; i < in.size(); ++i) {
// 			out[i] = exp(in[i]) / sum;
// 		}
// 	}
//
// 	void backward() override {}
// };
//
// class ReshapeLayer : public Layer {
//   public:
// 	array<size_t, 3> input_sizes;
// 	array<size_t, 3> output_sizes;
//
// 	ReshapeLayer(const array<size_t, 3> &input_dims, const array<size_t, 3> &output_dims) {
// 		this->input_sizes = input_dims;
// 		this->output_sizes = output_dims;
//
// 		if (input_sizes[0] * input_sizes[1] * input_sizes[2] != output_sizes[0] * output_sizes[1] * output_sizes[2]) {
// 			throw invalid_argument("Input and output don't have the same number of elements");
// 		}
// 	}
//
// 	void forward() override { output->data().assign(input->data().begin(), input->data().end()); }
//
// 	void backward() override {};
// };

class CNN {
  public:
	std::vector<Tensor<float, 3> *> buffers;
	std::vector<Layer *> layers;

	CNN(Tensor<float, 3> *input_data) { buffers.push_back(input_data); }

	template <typename LayerType> void add_layer(LayerType *layer, Tensor<float, 3> *output) {
		layer->input = buffers.back();
		buffers.push_back(output);
		layer->output = output;
		layers.push_back(layer);
	}

	void add_convolutional_layer(const array<Index, 2> &kernel_dims, size_t filters, size_t stride) {
		ConvolutionalLayer *layer = new ConvolutionalLayer(buffers.back()->dimensions(), kernel_dims, filters, stride);
		Tensor<float, 3> *output = new Tensor<float, 3>(layer->output_sizes[0], layer->output_sizes[1], layer->output_sizes[2]);

		this->add_layer(layer, output);
	}

	// void add_pooling_layer(const array<size_t, 2> &kernel_dims, size_t stride) {
	// 	MaxPoolingLayer *layer = new MaxPoolingLayer(buffers.back()->get_sizes(), kernel_dims, stride);
	// 	Matrix<3, float> *output = new Matrix<3, float>(layer->output_sizes);
	//
	// 	this->add_layer(layer, output);
	// }
	//
	// void add_fully_connected_layer(size_t output_dim) {
	// 	FullyConnectedLayer *layer = new FullyConnectedLayer(buffers.back()->get_sizes()[1], output_dim);
	// 	Matrix<3, float> *output = new Matrix<3, float>(1, output_dim, 1);
	//
	// 	this->add_layer(layer, output);
	// }
	//
	// void add_relu_layer() {
	// 	ReLULayer *layer = new ReLULayer();
	// 	Matrix<3, float> *output = new Matrix<3, float>(buffers.back()->get_sizes());
	//
	// 	this->add_layer(layer, output);
	// }
	//
	// void add_softmax_layer() {
	// 	SoftMaxLayer *layer = new SoftMaxLayer(buffers.back()->get_sizes());
	// 	Matrix<3, float> *output = new Matrix<3, float>(buffers.back()->get_sizes());
	//
	// 	this->add_layer(layer, output);
	// }
	//
	// void add_reshape_layer(const array<size_t, 3> &output_dims) {
	// 	ReshapeLayer *layer = new ReshapeLayer(buffers.back()->get_sizes(), output_dims);
	// 	Matrix<3, float> *output = new Matrix<3, float>(layer->output_sizes);
	//
	// 	this->add_layer(layer, output);
	// }
	//
	// void add_reshape_layer() {
	// 	array<size_t, 3> input_sizes = buffers.back()->get_sizes();
	// 	add_reshape_layer(array<size_t, 3>{1, input_sizes[0] * input_sizes[1] * input_sizes[2], 1});
	// }

	void forward() {
		for (auto &layer : layers) {
			layer->forward();
		}
	}

	void print_buffers() {
		for (auto it = buffers.begin(); it != buffers.end(); it++) {
			std::cout << (*it) << std::endl;
		}
	}
};
