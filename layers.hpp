#include "tensor_print.hpp"
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>
#define EIGEN_DONT_PARALLELIZE false

using namespace Eigen;
using std::cout, std::endl, std::vector;

class Layer {
  public:
	Tensor<float, 3> *input = nullptr;
	Tensor<float, 3> *output = nullptr;

	virtual void attach_network(vector<Tensor<float, 3> *> *buffers_ptr) {
		this->input = buffers_ptr->back();
		Tensor<float, 3> *output = new Tensor<float, 3>(input->dimensions());
		this->output = output;
		buffers_ptr->push_back(output);
	};

	virtual void forward() = 0;
	virtual void backward() = 0;

	virtual ~Layer() = default;
};

class ConvolutionalLayer : public Layer {
  public:
	vector<Index> input_indices;
	MatrixXf kernel;
	MatrixXf biases;

	array<Index, 3> input_sizes;
	array<Index, 3> output_sizes;
	array<Index, 2> kernel_sizes;

	Index stride;
	Index filters;
	Index channels;

	ConvolutionalLayer(vector<Tensor<float, 3> *> *buffers_ptr, const array<Index, 2> &kernel_dims, Index filters, Index stride) {
		this->input_sizes = buffers_ptr->back()->dimensions();
		this->kernel_sizes = kernel_dims;
		this->output_sizes = {filters, (input_sizes[1] - kernel_dims[0]) / stride + 1,
							  (input_sizes[2] - kernel_dims[1]) / stride + 1};

		this->stride = stride;
		this->filters = filters;
		this->channels = input_sizes[0];

		this->C = channels;
		this->K = filters;
		this->R = kernel_sizes[0];
		this->S = kernel_sizes[1];

		this->P = output_sizes[1];
		this->Q = output_sizes[2];

		this->H = input_sizes[1];
		this->W = input_sizes[2];

		biases = MatrixXf::Random(K, P * Q) / 2.0f;
		kernel = MatrixXf::Random(K, R * S * C) / 2.0f;

		attach_network(buffers_ptr);
		reindex_input();
	}

	void attach_network(vector<Tensor<float, 3> *> *buffers_ptr) override {
		this->input = buffers_ptr->back();
		Tensor<float, 3> *output = new Tensor<float, 3>(output_sizes);
		this->output = output;
		buffers_ptr->push_back(output);
	}

	void forward() override {
		using namespace std::chrono;

		MatrixXf input_reshaped(C * R * S, P * Q);
		float *in_ptr = input->data();
		float *reshaped_ptr = input_reshaped.data();

		for (Index i = 0; i < (C * R * S) * (P * Q); ++i) {
			reshaped_ptr[i] = in_ptr[input_indices[i]];
		}
		auto time_1 = high_resolution_clock::now();
		MatrixXf result = kernel * input_reshaped + biases; // shape = (K, PQ)
		auto time_2 = high_resolution_clock::now();
		cout << duration_cast<milliseconds>(time_2 - time_1).count() << " ms \n";

		for (Index f = 0; f < K; f++) {
			for (Index i = 0; i < P * Q; ++i) {
				(*output)(f, i / Q, i % Q) = result(f, i);
			}
		}
	}

	void backward() override {}

  private:
	Index C, R, S, P, Q, H, W, K;

	void reindex_input() {

		/* old shape = C x H x W
		new shape = CRS x PQ

		input_new(a, b) = input_old(input_indices[a*P*Q + b]) */

		input_indices.resize(C * R * S * P * Q);
		for (Index a = 0; a < C * R * S; a++) {
			for (Index b = 0; b < P * Q; b++) {

				Index current_channel = a / (R * S);

				Index kernel_r = b / Q;
				Index kernel_c = b % Q;

				Index value_r = (a % (R * S)) / R;
				Index value_c = (a % (R * S)) % R;

				// cout << endl << a << ", " << b << ": " << endl;

				input_indices[b * S * R * C + a] =
					(kernel_c * stride + value_c) * H * C + (kernel_r * stride + value_r) * C + current_channel;
				// WARNING: In memory matrices are column-major

				// cout << current_channel << ", row: " << kernel_r << ", " << value_r << ", column: " << kernel_c << ", " <<
				// value_c
				// 	 << "   " << input_indices[a * P * Q + b] << endl;
			}
		}
	}
};

class MaxPoolingLayer : public Layer {
  public:
	array<Index, 3> input_sizes;
	array<Index, 2> kernel_sizes;
	array<Index, 3> output_sizes;

	Index stride;

	MaxPoolingLayer(vector<Tensor<float, 3> *> *buffers_ptr, const array<Index, 2> &kernel_dims, Index stride) {
		this->input_sizes = buffers_ptr->back()->dimensions();
		this->kernel_sizes = kernel_dims;
		this->stride = stride;

		this->output_sizes = {input_sizes[0], (input_sizes[1] - kernel_sizes[0]) / stride + 1,
							  (input_sizes[2] - kernel_sizes[1]) / stride + 1};

		attach_network(buffers_ptr);
	}

	void attach_network(vector<Tensor<float, 3> *> *buffers_ptr) override {
		this->input = buffers_ptr->back();
		Tensor<float, 3> *output = new Tensor<float, 3>(output_sizes);
		this->output = output;
		buffers_ptr->push_back(output);
	}

	void forward() override {
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

	void backward() override {}
};

class FullyConnectedLayer : public Layer {
  public:
	MatrixXf weights;
	VectorXf biases;

	Index input_size;
	Index output_size;

	FullyConnectedLayer(vector<Tensor<float, 3> *> *buffers_ptr, Index output_dim) {
		weights = MatrixXf::Random(output_dim, buffers_ptr->back()->dimensions()[2]);
		biases = VectorXf::Random(output_dim);

		this->input_size = buffers_ptr->back()->dimensions()[2];
		this->output_size = output_dim;

		attach_network(buffers_ptr);
	}

	void attach_network(vector<Tensor<float, 3> *> *buffers_ptr) override {
		this->input = buffers_ptr->back();
		Tensor<float, 3> *output = new Tensor<float, 3>(1, 1, output_size);
		this->output = output;
		buffers_ptr->push_back(output);
	}

	void forward() override {
		VectorXf in_vec(input_size);
		for (int i = 0; i < input_size; ++i) {
			in_vec(i) = (*input)(0, 0, i);
		}

		VectorXf out_vec = weights * in_vec + biases;

		for (int i = 0; i < output_size; ++i) {
			(*output)(0, 0, i) = out_vec(i);
		}
	}

	void backward() override {}
};

class ReLULayer : public Layer {
  public:
	ReLULayer(vector<Tensor<float, 3> *> *buffers_ptr) { attach_network(buffers_ptr); };

	void forward() override {
		(*output) = (*input).unaryExpr([](float x) { return std::max(0.0f, x); });
	}

	void backward() override {}
};

class SoftMaxLayer : public Layer {
  public:
	Index input_size;

	SoftMaxLayer(vector<Tensor<float, 3> *> *buffers_ptr) {
		if (buffers_ptr->back()->dimensions()[0] != 1 || buffers_ptr->back()->dimensions()[1] != 1) {
			const array<Index, 3> dims = buffers_ptr->back()->dimensions();

			throw std::invalid_argument(
				"SoftMax works EXCLUSIVELY after fully-connected layers, but the shape of input layer is (" +
				std::to_string(dims[0]) + ", " + std::to_string(dims[1]) + ", " + std::to_string(dims[2]) + ")");
		}
		this->input_size = buffers_ptr->back()->dimensions()[2];

		attach_network(buffers_ptr);
	};

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

	ReshapeLayer(vector<Tensor<float, 3> *> *buffers_ptr, const array<Index, 3> &output_dims) {
		this->input_sizes = buffers_ptr->back()->dimensions();
		this->output_sizes = output_dims;

		if (input_sizes[0] * input_sizes[1] * input_sizes[2] != output_sizes[0] * output_sizes[1] * output_sizes[2]) {
			throw std::invalid_argument("Input and output don't have the same number of elements");
		}

		attach_network(buffers_ptr);
	}

	void attach_network(vector<Tensor<float, 3> *> *buffers_ptr) override {
		this->input = buffers_ptr->back();
		Tensor<float, 3> *output = new Tensor<float, 3>(output_sizes);
		this->output = output;
		buffers_ptr->push_back(output);
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
	vector<Tensor<float, 3> *> buffers;
	vector<Layer *> layers;

	CNN(Tensor<float, 3> *input_data) {
		srand((unsigned int)time(0));
		buffers.push_back(input_data);
	}

	void add_convolutional_layer(const array<Index, 2> &kernel_dims, Index filters, Index stride) {
		ConvolutionalLayer *layer = new ConvolutionalLayer(&buffers, kernel_dims, filters, stride);
		layers.push_back(layer);
	}

	void add_maxpooling_layer(const array<Index, 2> &kernel_dims, Index stride) {
		MaxPoolingLayer *layer = new MaxPoolingLayer(&buffers, kernel_dims, stride);
		layers.push_back(layer);
	}

	void add_fully_connected_layer(Index output_dim) {
		FullyConnectedLayer *layer = new FullyConnectedLayer(&buffers, output_dim);
		layers.push_back(layer);
	}

	void add_relu_layer() {
		ReLULayer *layer = new ReLULayer(&buffers);
		layers.push_back(layer);
	}

	void add_softmax_layer() {
		SoftMaxLayer *layer = new SoftMaxLayer(&buffers);
		layers.push_back(layer);
	}

	void add_reshape_layer(const array<Index, 3> &output_dims) {
		ReshapeLayer *layer = new ReshapeLayer(&buffers, output_dims);
		layers.push_back(layer);
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
