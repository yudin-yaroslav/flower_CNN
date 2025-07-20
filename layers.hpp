#include "dataset.hpp"
#include "tensor_print.hpp"
#include <iostream>
#include <stdexcept>
#include <typeinfo>
#include <vector>

#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>
#define EIGEN_DONT_PARALLELIZE false

using namespace Eigen;
using std::cout, std::endl, std::vector;

class Layer {
  public:
	Tensor<float, 3> *input_data = nullptr;
	Tensor<float, 3> *output_data = nullptr;

	Tensor<float, 3> *input_gradient = nullptr;
	Tensor<float, 3> *output_gradient = nullptr;

	virtual void attach_network(vector<Tensor<float, 3> *> *buffers_ptr, vector<Tensor<float, 3> *> *gradients_ptr) {
		this->input_data = buffers_ptr->back();
		this->input_gradient = gradients_ptr->back();

		Tensor<float, 3> *output_data = new Tensor<float, 3>(input_data->dimensions());
		this->output_data = output_data;
		buffers_ptr->push_back(output_data);

		Tensor<float, 3> *output_gradient = new Tensor<float, 3>(output_data->dimensions());
		this->output_gradient = output_gradient;
		gradients_ptr->push_back(output_gradient);
	};

	virtual void forward() = 0;
	virtual void backward(float learning_rate) = 0;

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

	ConvolutionalLayer(vector<Tensor<float, 3> *> *buffers_ptr, vector<Tensor<float, 3> *> *gradients_ptr,
					   const array<Index, 2> &kernel_dims, Index filters, Index stride) {
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

		biases = MatrixXf::Zero(K, P * Q) / 2.0f;
		kernel = MatrixXf::Zero(K, R * S * C) / 2.0f;

		attach_network(buffers_ptr, gradients_ptr);
		reindex_input();
	}

	void attach_network(vector<Tensor<float, 3> *> *buffers_ptr, vector<Tensor<float, 3> *> *gradients_ptr) override {
		this->input_data = buffers_ptr->back();
		this->input_gradient = gradients_ptr->back();

		Tensor<float, 3> *output = new Tensor<float, 3>(output_sizes);
		this->output_data = output;
		buffers_ptr->push_back(output);

		Tensor<float, 3> *output_gradient = new Tensor<float, 3>(output_data->dimensions());
		this->output_gradient = output_gradient;
		gradients_ptr->push_back(output_gradient);
	}

	void forward() override {
		using namespace std::chrono;

		MatrixXf input_reshaped(C * R * S, P * Q);
		float *in_ptr = input_data->data();
		float *reshaped_ptr = input_reshaped.data();

		for (Index i = 0; i < (C * R * S) * (P * Q); ++i) {
			reshaped_ptr[i] = in_ptr[input_indices[i]];
		}
		MatrixXf result = kernel * input_reshaped + biases; // shape = (K, PQ)

		for (Index f = 0; f < K; f++) {
			for (Index i = 0; i < P * Q; ++i) {
				(*output_data)(f, i / Q, i % Q) = result(f, i);
			}
		}
	}

	void backward(float learning_rate) override {}

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

				// WARNING: In memory matrices are column-major
				input_indices[b * S * R * C + a] =
					(kernel_c * stride + value_c) * H * C + (kernel_r * stride + value_r) * C + current_channel;
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

	MaxPoolingLayer(vector<Tensor<float, 3> *> *buffers_ptr, vector<Tensor<float, 3> *> *gradients_ptr,
					const array<Index, 2> &kernel_dims, Index stride) {
		this->input_sizes = buffers_ptr->back()->dimensions();
		this->kernel_sizes = kernel_dims;
		this->stride = stride;

		this->output_sizes = {input_sizes[0], (input_sizes[1] - kernel_sizes[0]) / stride + 1,
							  (input_sizes[2] - kernel_sizes[1]) / stride + 1};

		attach_network(buffers_ptr, gradients_ptr);
	}

	void attach_network(vector<Tensor<float, 3> *> *buffers_ptr, vector<Tensor<float, 3> *> *gradients_ptr) override {
		this->input_data = buffers_ptr->back();
		this->input_gradient = gradients_ptr->back();

		Tensor<float, 3> *output = new Tensor<float, 3>(output_sizes);
		this->output_data = output;
		buffers_ptr->push_back(output);

		Tensor<float, 3> *output_gradient = new Tensor<float, 3>(output_data->dimensions());
		this->output_gradient = output_gradient;
		gradients_ptr->push_back(output_gradient);
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

							max_val = std::max(max_val, (*input_data)(c, in_i, in_j));
						}
					}
					(*output_data)(c, i, j) = max_val;
				}
			}
		}
	}

	void backward(float learning_rate) override {
		(*input_gradient).setZero();

		for (Index c = 0; c < output_sizes[0]; ++c) {
			for (Index i = 0; i < output_sizes[1]; ++i) {
				for (Index j = 0; j < output_sizes[2]; ++j) {

					for (Index x = 0; x < kernel_sizes[0]; ++x) {
						for (Index y = 0; y < kernel_sizes[1]; ++y) {
							Index in_i = i * stride + x;
							Index in_j = j * stride + y;

							if ((*output_data)(c, i, j) == (*input_data)(c, in_i, in_j)) {
								(*input_gradient)(c, in_i, in_j) = (*output_gradient)(c, i, j);
								goto break_label;
							}
						}
					}

				break_label:;
				}
			}
		}
	}
};

class FullyConnectedLayer : public Layer {
  public:
	MatrixXf weights;
	VectorXf biases;

	MatrixXf weights_gradients;
	VectorXf biases_gradients;

	Index input_size;
	Index output_size;

	FullyConnectedLayer(vector<Tensor<float, 3> *> *buffers_ptr, vector<Tensor<float, 3> *> *gradients_ptr, Index output_dim) {
		weights = MatrixXf::Zero(output_dim, buffers_ptr->back()->dimensions()[2]);
		biases = VectorXf::Zero(output_dim);

		weights_gradients = MatrixXf::Zero(output_dim, buffers_ptr->back()->dimensions()[2]);
		biases_gradients = VectorXf::Zero(output_dim);

		this->input_size = buffers_ptr->back()->dimensions()[2];
		this->output_size = output_dim;

		attach_network(buffers_ptr, gradients_ptr);
	}

	void attach_network(vector<Tensor<float, 3> *> *buffers_ptr, vector<Tensor<float, 3> *> *gradients_ptr) override {
		this->input_data = buffers_ptr->back();
		this->input_gradient = gradients_ptr->back();

		Tensor<float, 3> *output = new Tensor<float, 3>(1, 1, output_size);
		this->output_data = output;
		buffers_ptr->push_back(output);

		Tensor<float, 3> *output_gradient = new Tensor<float, 3>(output_data->dimensions());
		this->output_gradient = output_gradient;
		gradients_ptr->push_back(output_gradient);
	}

	void update_gradients() {
		// input gradient
		VectorXf output_gradients_vec(output_size);
		for (int i = 0; i < output_size; ++i) {
			output_gradients_vec(i) = (*output_gradient)(0, 0, i);
		}
		VectorXf input_gradient_vec = weights.transpose() * output_gradients_vec;

		std::copy(input_gradient_vec.data(), input_gradient_vec.data() + input_gradient_vec.size(), input_gradient->data());

		// weight gradients
		VectorXf input_data_vec(input_size);
		for (int i = 0; i < input_size; ++i) {
			input_data_vec(i) = (*input_data)(0, 0, i);
		}
		weights_gradients = output_gradients_vec * input_data_vec.transpose();

		// bias gradients
		biases_gradients = output_gradients_vec;
	}

	void forward() override {
		VectorXf in_vec(input_size);
		for (int i = 0; i < input_size; ++i) {
			in_vec(i) = (*input_data)(0, 0, i);
		}

		VectorXf out_vec = weights * in_vec + biases;

		for (int i = 0; i < output_size; ++i) {
			(*output_data)(0, 0, i) = out_vec(i);
		}
	}

	void backward(float learning_rate) override {
		update_gradients();

		weights -= learning_rate * weights_gradients;
		biases -= learning_rate * biases_gradients;
	}
};

class ReLULayer : public Layer {
  public:
	array<Index, 3> input_sizes;

	ReLULayer(vector<Tensor<float, 3> *> *buffers_ptr, vector<Tensor<float, 3> *> *gradients_ptr) {
		this->input_sizes = buffers_ptr->back()->dimensions();

		attach_network(buffers_ptr, gradients_ptr);
	};

	void forward() override {
		(*output_data) = (*input_data).unaryExpr([](float x) { return std::max(0.0f, x); });
	}

	void backward(float learning_rate) override {
		for (Index i = 0; i < input_sizes[0]; ++i) {
			for (Index j = 0; j < input_sizes[1]; ++j) {
				for (Index k = 0; k < input_sizes[2]; ++k) {
					if ((*output_data)(i, j, k) == 0) {
						(*input_gradient)(i, j, k) = 0;
					} else {
						(*input_gradient)(i, j, k) = (*output_gradient)(i, j, k);
					}
				}
			}
		}
	}
};

class SoftMaxLayer : public Layer {
  public:
	Index input_size;

	SoftMaxLayer(vector<Tensor<float, 3> *> *buffers_ptr, vector<Tensor<float, 3> *> *gradients_ptr) {
		const array<Index, 3> dims = buffers_ptr->back()->dimensions();

		if (buffers_ptr->back()->dimensions()[0] != 1 || buffers_ptr->back()->dimensions()[1] != 1) {
			throw std::invalid_argument(
				"SoftMax works EXCLUSIVELY after fully-connected layers, but the shape of input layer is (" +
				std::to_string(dims[0]) + ", " + std::to_string(dims[1]) + ", " + std::to_string(dims[2]) + ")");
		}
		this->input_size = dims[2];

		attach_network(buffers_ptr, gradients_ptr);
	};

	void forward() override {
		Tensor<float, 0> max_tensor = (*input_data).maximum();
		float max_value = max_tensor(0);

		float sum = 0.0f;
		for (Index i = 0; i < input_size; i++) {
			sum += exp((*input_data)(0, 0, i) - max_value);
		}

		for (Index i = 0; i < input_size; ++i) {
			(*output_data)(0, 0, i) = exp((*input_data)(0, 0, i) - max_value) / sum;
		}
	}

	void backward(float learning_rate) override {
		for (Index i = 0; i < input_size; i++) {
			float sum = 0.0f;
			for (Index j = 0; j < input_size; j++) {
				if (j == i) {
					sum += (*output_gradient)(0, 0, j) * (*output_data)(0, 0, i) * (1 - (*output_data)(0, 0, i));
				} else {
					sum += -(*output_gradient)(0, 0, j) * (*output_data)(0, 0, i) * (*output_data)(0, 0, j);
				}
			}
			(*input_gradient)(0, 0, i) = sum;
		}
	}
};

class ReshapeLayer : public Layer {
  public:
	array<Index, 3> input_sizes;
	array<Index, 3> output_sizes;

	Index length;

	ReshapeLayer(vector<Tensor<float, 3> *> *buffers_ptr, vector<Tensor<float, 3> *> *gradients_ptr,
				 const array<Index, 3> &output_dims) {
		this->input_sizes = buffers_ptr->back()->dimensions();
		this->output_sizes = output_dims;
		this->length = input_sizes[0] * input_sizes[1] * input_sizes[2];

		if (length != output_sizes[0] * output_sizes[1] * output_sizes[2]) {
			throw std::invalid_argument("Input and output don't have the same number of elements");
		}

		attach_network(buffers_ptr, gradients_ptr);
	}

	void attach_network(vector<Tensor<float, 3> *> *buffers_ptr, vector<Tensor<float, 3> *> *gradients_ptr) override {
		this->input_data = buffers_ptr->back();
		this->input_gradient = gradients_ptr->back();

		Tensor<float, 3> *output = new Tensor<float, 3>(output_sizes);
		this->output_data = output;
		buffers_ptr->push_back(output);

		Tensor<float, 3> *output_gradient = new Tensor<float, 3>(output_data->dimensions());
		this->output_gradient = output_gradient;
		gradients_ptr->push_back(output_gradient);
	}

	void forward() override {
		for (Index i = 0; i < input_sizes[0]; ++i) {
			for (Index j = 0; j < input_sizes[1]; ++j) {
				for (Index k = 0; k < input_sizes[2]; ++k) {
					Index flattten_index = i * (input_sizes[1] * input_sizes[2]) + j * input_sizes[2] + k;
					(*output_data)(0, 0, flattten_index) = (*input_data)(i, j, k);
				}
			}
		}
	}

	void backward(float learning_rate = 0.0f) override {
		for (Index i = 0; i < input_sizes[0]; ++i) {
			for (Index j = 0; j < input_sizes[1]; ++j) {
				for (Index k = 0; k < input_sizes[2]; ++k) {
					Index flattten_index = i * (input_sizes[1] * input_sizes[2]) + j * input_sizes[2] + k;
					(*input_gradient)(i, j, k) = (*output_gradient)(0, 0, flattten_index);
				}
			}
		}
	};
};

class CNN {
  public:
	vector<Tensor<float, 3> *> data_buffer;
	vector<Tensor<float, 3> *> gradient_buffer;

	vector<Layer *> layers;
	int number_of_layers;

	float learning_rate;
	int number_of_predictions;

	vector<float> predictions;

	CNN(Tensor<float, 3> *input_data, float learning_rate) {
		srand((unsigned int)time(0));
		Tensor<float, 3> *input_gradient_dummy = new Tensor<float, 3>(input_data->dimensions());

		data_buffer.push_back(input_data);
		gradient_buffer.push_back(input_gradient_dummy);

		this->learning_rate = learning_rate;
		number_of_layers = 0;
	}

	CNN(const array<Index, 3> image_dims, float learning_rate) {
		srand((unsigned int)time(0));
		Tensor<float, 3> *input_data_dummy = new Tensor<float, 3>(image_dims);
		Tensor<float, 3> *input_gradient_dummy = new Tensor<float, 3>(image_dims);

		data_buffer.push_back(input_data_dummy);
		gradient_buffer.push_back(input_gradient_dummy);

		this->learning_rate = learning_rate;
		number_of_layers = 0;
	}

	~CNN() {
		for (auto layer : layers) {
			delete layer;
		}
		layers.clear();
	}

	void set_input_data(Tensor<float, 3> *input_data) {
		if (data_buffer.size() == 0) {
			data_buffer.push_back(input_data);
		}
		data_buffer[0] = input_data;
	}

	void add_convolutional_layer(const array<Index, 2> &kernel_dims, Index filters, Index stride) {
		ConvolutionalLayer *layer = new ConvolutionalLayer(&data_buffer, &gradient_buffer, kernel_dims, filters, stride);
		layers.push_back(layer);
		number_of_layers++;
	}

	void add_maxpooling_layer(const array<Index, 2> &kernel_dims, Index stride) {
		MaxPoolingLayer *layer = new MaxPoolingLayer(&data_buffer, &gradient_buffer, kernel_dims, stride);
		layers.push_back(layer);
		number_of_layers++;
	}

	void add_fully_connected_layer(Index output_dim) {
		FullyConnectedLayer *layer = new FullyConnectedLayer(&data_buffer, &gradient_buffer, output_dim);
		layers.push_back(layer);
		number_of_layers++;
	}

	void add_relu_layer() {
		ReLULayer *layer = new ReLULayer(&data_buffer, &gradient_buffer);
		layers.push_back(layer);
		number_of_layers++;
	}

	void add_softmax_layer() {
		SoftMaxLayer *layer = new SoftMaxLayer(&data_buffer, &gradient_buffer);

		this->number_of_predictions = layer->input_size;
		this->predictions.resize(number_of_predictions);

		layers.push_back(layer);
		number_of_layers++;
	}

	void add_reshape_layer(const array<Index, 3> &output_dims) {
		ReshapeLayer *layer = new ReshapeLayer(&data_buffer, &gradient_buffer, output_dims);
		layers.push_back(layer);
		number_of_layers++;
	}

	void add_flatten_layer() {
		array<Index, 3> input_sizes = data_buffer.back()->dimensions();
		add_reshape_layer(array<Index, 3>{1, 1, input_sizes[0] * input_sizes[1] * input_sizes[2]});
	}

	float get_mean_squared_error(int correct_prediction) {
		float mean_squared_error = 0.0f;

		for (int i = 0; i < number_of_predictions; i++) {
			if (i == correct_prediction) {
				mean_squared_error += (1 - predictions[i]) * (1 - predictions[i]);
			} else {
				mean_squared_error += predictions[i] * predictions[i];
			}
		}

		mean_squared_error /= number_of_predictions;

		return mean_squared_error;
	}

	void forward() {
		for (int i = 0; i < number_of_layers; i++) {
			layers[i]->forward();
		}

		if (dynamic_cast<SoftMaxLayer *>(layers.back()) != nullptr) {
			for (int i = 0; i < number_of_predictions; i++) {
				predictions[i] = (*data_buffer.back())(0, 0, i);
			}
		}
	}

	void backward(int correct_prediction) {
		if (dynamic_cast<SoftMaxLayer *>(layers.back()) != nullptr) {
			// We shall create gradient for softmax first
			for (int i = 0; i < number_of_predictions; i++) {
				if (i == correct_prediction) {
					(*gradient_buffer[number_of_layers])(0, 0, i) = (predictions[i] - 1) * 2 / number_of_predictions;
				} else {
					(*gradient_buffer[number_of_layers])(0, 0, i) = (predictions[i]) * 2 / number_of_predictions;
				}
			}
		}

		for (int i = number_of_layers - 1; i >= 0; i--) {
			layers[i]->backward(learning_rate);
		}
	}

	void print_buffers() {
		for (auto it = data_buffer.begin(); it != data_buffer.end(); it++) {
			cout << (*it) << endl;
		}
	}
};
