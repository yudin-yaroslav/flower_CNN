#include <cstddef>
#include <iostream>
#include <vector>

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

#define EIGEN_USE_THREADS

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
	Tensor<float, 4> kernel; // [filters, channels, kernel_h, kernel_w]
	Tensor<float, 3> biases; // [filters, out_h, out_w]

	array<Index, 3> input_sizes;
	array<Index, 3> output_sizes;
	array<Index, 2> kernel_sizes;

	Index stride;
	Index filters;
	Index channels;

	ConvolutionalLayer(const array<Index, 3> &input_dims, const array<Index, 2> &kernel_dims, Index filters, Index stride)
		: stride(stride), filters(filters), channels(input_dims[0]) {

		input_sizes = input_dims;
		kernel_sizes = kernel_dims;

		output_sizes = {filters, (input_dims[1] - kernel_dims[0]) / stride + 1, (input_dims[2] - kernel_dims[1]) / stride + 1};

		kernel = Tensor<float, 4>(filters, channels, kernel_dims[0], kernel_dims[1]);
		kernel.setRandom();

		biases = Tensor<float, 3>(filters, output_sizes[1], output_sizes[2]);
		biases.setRandom();
	}

	void forward() override {
		for (Index f = 0; f < filters; ++f) {
			MatrixXf out_sub = MatrixXf::Zero(output_sizes[1], output_sizes[2]);

			for (Index c = 0; c < channels; ++c) {
				for (Index x = 0; x < output_sizes[1]; ++x) {
					for (Index y = 0; y < output_sizes[2]; ++y) {
						Index in_x = x * stride;
						Index in_y = y * stride;

						float acc = 0.0f;
						for (Index i = 0; i < kernel_sizes[0]; ++i) {
							for (Index j = 0; j < kernel_sizes[1]; ++j) {
								acc += (*input)(c, in_x + i, in_y + j) * kernel(f, c, i, j);
							}
						}
						out_sub(x, y) += acc;
					}
				}
			}

			float *base_ptr = output->data() + f * output_sizes[1] * output_sizes[2];
			Map<Matrix<float, Dynamic, Dynamic, RowMajor>> out_map(base_ptr, output_sizes[1], output_sizes[2]);
			Map<Matrix<float, Dynamic, Dynamic, RowMajor>> bias_map(biases.data() + f * output_sizes[1] * output_sizes[2],
																	output_sizes[1], output_sizes[2]);

			out_map = out_sub + bias_map;
		}
	}

	void backward() override {}
};
class MaxPoolingLayer : public Layer {
  public:
	array<Index, 3> input_sizes;
	array<Index, 2> kernel_sizes;
	array<Index, 3> output_sizes;

	Index stride;

	MaxPoolingLayer(const array<Index, 3> &input_dims, const array<Index, 2> &kernel_dims, Index stride) {
		this->input_sizes = input_dims;
		this->kernel_sizes = kernel_dims;
		this->stride = stride;

		this->output_sizes = {input_sizes[0], (input_sizes[1] - kernel_sizes[0]) / stride + 1,
							  (input_sizes[2] - kernel_sizes[1]) / stride + 1};
	}

	void forward() {
		for (Index c = 0; c < output_sizes[0]; ++c) {
			for (Index i = 0; i < output_sizes[1]; ++i) {
				for (Index j = 0; j < output_sizes[2]; ++j) {

					float max_val = -INFINITY;
					for (Index x = 0; x < kernel_sizes[0]; ++x) {
						for (Index y = 0; y < kernel_sizes[1]; ++y) {
							Index in_i = i * stride + x;
							Index in_j = j * stride + y;

							max_val = std::max(max_val, (*input)(c, in_i, in_j));
						}
					}
					(*output)(c, i, j) = max_val;
				}
			}
		}
	}

	void backward() {}
};

class FullyConnectedLayer : public Layer {
  public:
	MatrixXf weights;

	Index input_size;
	Index output_size;

	FullyConnectedLayer(Index input_dim, Index output_dim) {
		weights = MatrixXf(output_dim, input_dim);
		weights.setRandom();

		this->input_size = input_dim;
		this->output_size = output_dim;
	}

	void forward() override {
		VectorXf in_vec(input_size);
		for (int i = 0; i < input_size; ++i) {
			in_vec(i) = (*input)(0, 0, i);
		}

		VectorXf out_vec = weights * in_vec;

		for (int i = 0; i < output_size; ++i) {
			(*output)(0, 0, i) = out_vec(i);
		}
	}

	void backward() override {}
};

class ReLULayer : public Layer {
  public:
	ReLULayer() {};

	void forward() override {
		(*output) = (*input).unaryExpr([](float x) { return std::max(0.0f, x); });
	}

	void backward() override {}
};

class SoftMaxLayer : public Layer {
	// NOTE: SoftMax works EXCLUSIVELY after fully-connected layers.

  public:
	Index input_size;

	SoftMaxLayer(Index input_dim) { this->input_size = input_dim; };

	void forward() override {
		Tensor<float, 0> max_tensor = (*input).maximum();
		float max_value = max_tensor(0);

		float sum = 0.0f;
		for (Index i = 0; i < input_size; i++) {
			sum += exp((*input)(0, 0, i) - max_value);
		}

		for (Index i = 0; i < input_size; ++i) {
			(*output)(0, 0, i) = exp((*input)(0, 0, i) - max_value) / sum;
		}
	}

	void backward() override {}
};

class ReshapeLayer : public Layer {
  public:
	array<Index, 3> input_sizes;
	array<Index, 3> output_sizes;

	ReshapeLayer(const array<Index, 3> &input_dims, const array<Index, 3> &output_dims) {
		this->input_sizes = input_dims;
		this->output_sizes = output_dims;

		if (input_sizes[0] * input_sizes[1] * input_sizes[2] != output_sizes[0] * output_sizes[1] * output_sizes[2]) {
			throw std::invalid_argument("Input and output don't have the same number of elements");
		}
	}

	void forward() override {
		for (Index i = 0; i < input_sizes[0]; ++i) {
			for (Index j = 0; j < input_sizes[1]; ++j) {
				for (Index k = 0; k < input_sizes[2]; ++k) {
					Index flattten_index = i * (input_sizes[1] * input_sizes[2]) + j * input_sizes[2] + k;
					(*output)(0, 0, flattten_index) = (*input)(i, j, k);
				}
			}
		}
	}

	void backward() override {};
};

class CNN {
  public:
	std::vector<Tensor<float, 3> *> buffers;
	std::vector<Layer *> layers;

	CNN(Tensor<float, 3> *input_data) {
		srand((unsigned int)time(0));
		buffers.push_back(input_data);
	}

	template <typename LayerType> void add_layer(LayerType *layer, Tensor<float, 3> *output) {
		layer->input = buffers.back();
		buffers.push_back(output);
		layer->output = output;
		layers.push_back(layer);
	}

	void add_convolutional_layer(const array<Index, 2> &kernel_dims, Index filters, Index stride) {
		ConvolutionalLayer *layer = new ConvolutionalLayer(buffers.back()->dimensions(), kernel_dims, filters, stride);
		Tensor<float, 3> *output = new Tensor<float, 3>(layer->output_sizes[0], layer->output_sizes[1], layer->output_sizes[2]);

		this->add_layer(layer, output);
	}

	void add_maxpooling_layer(const array<Index, 2> &kernel_dims, Index stride) {
		MaxPoolingLayer *layer = new MaxPoolingLayer(buffers.back()->dimensions(), kernel_dims, stride);
		Tensor<float, 3> *output = new Tensor<float, 3>(layer->output_sizes[0], layer->output_sizes[1], layer->output_sizes[2]);

		this->add_layer(layer, output);
	}

	void add_fully_connected_layer(Index output_dim) {
		FullyConnectedLayer *layer = new FullyConnectedLayer(buffers.back()->dimensions()[2], output_dim);
		Tensor<float, 3> *output = new Tensor<float, 3>(1, 1, layer->output_size);

		this->add_layer(layer, output);
	}

	void add_relu_layer() {
		ReLULayer *layer = new ReLULayer();
		Tensor<float, 3> *output = new Tensor<float, 3>(buffers.back()->dimensions());

		this->add_layer(layer, output);
	}

	void add_softmax_layer() {
		SoftMaxLayer *layer = new SoftMaxLayer(buffers.back()->dimensions()[2]);
		Tensor<float, 3> *output = new Tensor<float, 3>(buffers.back()->dimensions());

		this->add_layer(layer, output);
	}

	void add_reshape_layer(const array<Index, 3> &output_dims) {
		ReshapeLayer *layer = new ReshapeLayer(buffers.back()->dimensions(), output_dims);
		Tensor<float, 3> *output = new Tensor<float, 3>(layer->output_sizes);

		this->add_layer(layer, output);
	}

	void add_flatten_layer() {
		array<Index, 3> input_sizes = buffers.back()->dimensions();
		add_reshape_layer(array<Index, 3>{1, 1, input_sizes[0] * input_sizes[1] * input_sizes[2]});
	}

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
