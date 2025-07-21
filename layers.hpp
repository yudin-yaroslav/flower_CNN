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

	Tensor<float, 3> *input_gradients = nullptr;
	Tensor<float, 3> *output_gradients = nullptr;

	virtual void attach_network(vector<Tensor<float, 3> *> *buffers_ptr, vector<Tensor<float, 3> *> *gradients_ptr) {
		this->input_data = buffers_ptr->back();
		this->input_gradients = gradients_ptr->back();

		Tensor<float, 3> *output_data = new Tensor<float, 3>(input_data->dimensions());
		this->output_data = output_data;
		buffers_ptr->push_back(output_data);

		Tensor<float, 3> *output_gradient = new Tensor<float, 3>(output_data->dimensions());
		this->output_gradients = output_gradient;
		gradients_ptr->push_back(output_gradient);
	};

	virtual void forward() = 0;
	virtual void backward(float learning_rate) = 0;

	virtual ~Layer() = default;
};

class ConvolutionalLayer : public Layer {
  public:
	vector<Index> input_reshaped_for_forw_indices;
	// vector<Index> input_reshaped_for_back_indices;

	vector<Index> out_grad_reshaped_for_input_indices;
	vector<Index> out_grad_reshaped_for_ker_indices;

	vector<Index> kernel_reshaped_indices;

	MatrixXf bias_gradients;
	MatrixXf kernel_gradients;

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

		this->stride = stride;
		this->filters = filters;
		this->channels = input_sizes[0];

		this->C = channels;
		this->K = filters;
		this->R = kernel_sizes[0];
		this->S = kernel_sizes[1];

		this->H = input_sizes[1];
		this->W = input_sizes[2];

		this->P = (H - R) / stride + 1;
		this->Q = (W - S) / stride + 1;

		this->H_pad = P + (R - 1) * 2;
		this->W_pad = Q + (S - 1) * 2;
		this->P_pad = H_pad - R + 1;
		this->Q_pad = W_pad - S + 1;

		this->output_sizes = {K, P, Q};

		biases = MatrixXf::Random(K, P * Q) / 1000.0f;
		kernel = MatrixXf::Random(K, R * S * C) / 1000.0f;

		bias_gradients = MatrixXf::Random(K, P * Q) / 1000.0f;
		kernel_gradients = MatrixXf::Random(K, R * S * C) / 1000.0f;

		attach_network(buffers_ptr, gradients_ptr);

		reindex_input_data_for_forw();
		reindex_output_gradient_for_input();
		reindex_output_gradient_for_kernel();
		reindex_kernels();
	}

	void attach_network(vector<Tensor<float, 3> *> *buffers_ptr, vector<Tensor<float, 3> *> *gradients_ptr) override {
		this->input_data = buffers_ptr->back();
		this->input_gradients = gradients_ptr->back();

		Tensor<float, 3> *output = new Tensor<float, 3>(output_sizes);
		this->output_data = output;
		buffers_ptr->push_back(output);

		Tensor<float, 3> *output_gradient = new Tensor<float, 3>(output_data->dimensions());
		this->output_gradients = output_gradient;
		gradients_ptr->push_back(output_gradient);
	}

	void update_input_gradients() {
		MatrixXf kernel_reshaped(C, K * R * S);
		float *kernel_ptr = kernel.data();
		float *kernel_reshaped_ptr = kernel_reshaped.data();

		for (Index i = 0; i < (C * K * R * S); ++i) {
			kernel_reshaped_ptr[i] = kernel_ptr[kernel_reshaped_indices[i]];
		}

		// cout << "\nKernel reshaped for input: " << endl;
		// PrintMatrixNumpy(kernel_reshaped, C, K * R * S);

		MatrixXf output_grad_reshaped_in(K * R * S, P_pad * Q_pad);
		float *out_ptr_in = output_gradients->data();
		float *out_reshaped_ptr_in = output_grad_reshaped_in.data();

		for (Index i = 0; i < (K * R * S) * (P_pad * Q_pad); ++i) {
			if (out_grad_reshaped_for_input_indices[i] == K * R * S) {
				out_reshaped_ptr_in[i] = 0.0f;
			} else {
				out_reshaped_ptr_in[i] = out_ptr_in[out_grad_reshaped_for_input_indices[i]];
			}
		}

		// cout << "\nOutput gradient reshaped for input: " << endl;
		// PrintMatrixNumpy(output_grad_reshaped_in, R * S * K, P_pad * Q_pad);

		MatrixXf input_gradient_mat = kernel_reshaped * output_grad_reshaped_in;
		// PrintMatrixNumpy(input_gradient_mat, C, H * W);

		for (Index channel = 0; channel < C; ++channel) {
			for (Index r = 0; r < H; ++r) {
				for (Index c = 0; c < W; ++c) {
					(*input_gradients)(channel, r, c) = input_gradient_mat(channel, r * W + c);
				}
			}
		}
	}

	void update_bias_gradients() {
		for (Index f = 0; f < K; f++) {
			for (Index r = 0; r < P; r++) {
				for (Index c = 0; c < Q; c++) {
					bias_gradients(f, r + c * P) = (*output_gradients)(f, r, c);
				}
			}
		}
	}

	void update_kernel_gradients() {
		MatrixXf output_grad_reshaped_ker(K, P * Q);
		float *out_ptr_ker = output_gradients->data();
		float *out_reshaped_ptr_ker = output_grad_reshaped_ker.data();

		for (Index i = 0; i < K * P * Q; ++i) {
			out_reshaped_ptr_ker[i] = out_ptr_ker[out_grad_reshaped_for_ker_indices[i]];
		}

		// cout << "\nOutput gradient reshaped for kernel: " << endl;
		// PrintMatrixNumpy(output_grad_reshaped_ker, K, P * Q);

		MatrixXf input_data_reshaped_ker(C * R * S, P * Q);
		float *in_ptr = input_data->data();
		float *in_reshaped_ptr = input_data_reshaped_ker.data();

		for (Index i = 0; i < C * R * S * P * Q; ++i) {
			// in_reshaped_ptr[i] = in_ptr[input_reshaped_for_back_indices[i]];
			in_reshaped_ptr[i] = in_ptr[input_reshaped_for_forw_indices[i]];
		}

		// cout << "\nInput data reshaped for kernel: " << endl;
		// PrintMatrixNumpy(input_data_reshaped_ker, C * R * S, P * Q);
		// cout << "\n\n";

		kernel_gradients = output_grad_reshaped_ker * input_data_reshaped_ker.transpose();
	}

	void forward() override {
		MatrixXf input_reshaped(C * R * S, P * Q);
		float *in_ptr = input_data->data();
		float *reshaped_ptr = input_reshaped.data();

		for (Index i = 0; i < (C * R * S) * (P * Q); ++i) {
			reshaped_ptr[i] = in_ptr[input_reshaped_for_forw_indices[i]];
		}
		MatrixXf result = kernel * input_reshaped + biases; // shape = (K, PQ)

		for (Index f = 0; f < K; f++) {
			for (Index i = 0; i < P * Q; ++i) {
				(*output_data)(f, i / Q, i % Q) = result(f, i);
			}
		}
	}

	void backward(float learning_rate) override {
		update_input_gradients();
		update_bias_gradients();
		update_kernel_gradients();

		biases -= learning_rate * bias_gradients;
		kernel -= learning_rate * kernel_gradients;
	}

  private:
	Index C, R, S, P, Q, H, W, K;
	Index H_pad, W_pad, P_pad, Q_pad;

	void reindex_input_data_for_forw() {
		input_reshaped_for_forw_indices.resize(C * R * S * P * Q);
		for (Index a = 0; a < C * R * S; a++) {
			for (Index b = 0; b < P * Q; b++) {

				Index current_channel = a / (R * S);

				Index kernel_r = b / Q;
				Index kernel_c = b % Q;

				Index value_r = (a % (R * S)) / R;
				Index value_c = (a % (R * S)) % R;

				Index pos_r = kernel_r * stride + value_r;
				Index pos_c = kernel_c * stride + value_c;

				// WARNING: In memory matrices are column-major
				input_reshaped_for_forw_indices[b * S * R * C + a] = pos_c * H * C + pos_r * C + current_channel;
			}
		}
	}

	void reindex_output_gradient_for_input() {
		if (P_pad != H || Q_pad != W) {
			throw std::logic_error("Stride is not implemented yet");
		}

		out_grad_reshaped_for_input_indices.resize(R * S * K * P_pad * Q_pad);
		for (Index a = 0; a < K * R * S; a++) {
			for (Index b = 0; b < P_pad * Q_pad; b++) {

				Index current_filter = a / (R * S);

				Index kernel_r = b / Q_pad;
				Index kernel_c = b % Q_pad;

				Index value_r = (a % (R * S)) / R;
				Index value_c = (a % (R * S)) % R;

				Index pos_r = kernel_r * stride + value_r;
				Index pos_c = kernel_c * stride + value_c;

				// WARNING: In memory matrices are column-majorc

				if (pos_r < R - 1 || pos_r >= P + (R - 1) || pos_c < S - 1 || pos_c >= Q + (S - 1)) {
					out_grad_reshaped_for_input_indices[b * S * R * K + a] = P * Q * K;
				} else {
					pos_r -= (R - 1);
					pos_c -= (S - 1);
					out_grad_reshaped_for_input_indices[b * S * R * K + a] = pos_c * P * K + pos_r * K + current_filter;
				}
			}
		}
	}

	void reindex_output_gradient_for_kernel() {
		out_grad_reshaped_for_ker_indices.resize(K * P * Q);
		for (Index a = 0; a < K; a++) {
			for (Index b = 0; b < P * Q; b++) {
				Index current_filter = a;

				Index pos_r = b / Q;
				Index pos_c = b % Q;

				out_grad_reshaped_for_ker_indices[b * K + a] = pos_c * Q * K + pos_r * K + current_filter;
			}
		}
	}

	void reindex_kernels() {
		kernel_reshaped_indices.resize(C * R * S * K);
		for (int a = 0; a < C; a++) {
			for (int b = 0; b < K * R * S; b++) {
				Index current_channel = a;
				Index current_filter = b / (R * S);

				Index kernel_index_old = b % (R * S);
				Index kernel_index_new = R * S - 1 - kernel_index_old;

				kernel_reshaped_indices[b * C + a] = current_filter + K * kernel_index_new + K * R * S * current_channel;
			}
		}
	}

	// void reindex_input_data_for_back() {
	// 	input_reshaped_for_back_indices.resize(C * R * S * P * Q);
	//
	// 	for (Index a = 0; a < C * R * S; a++) {
	// 		for (Index b = 0; b < P * Q; b++) {
	//
	// 			Index current_channel = a / (R * S);
	//
	// 			Index kernel_r = b / Q;
	// 			Index kernel_c = b % Q;
	//
	// 			Index value_r = (a % (R * S)) / R;
	// 			Index value_c = (a % (R * S)) % R;
	//
	// 			Index pos_r = kernel_r * stride + value_r;
	// 			Index pos_c = kernel_c * stride + value_c;
	//
	// 			// WARNING: In memory matrices are column-major
	// 			input_reshaped_for_forw_indices[b * S * R * C + a] = pos_c * H * C + pos_r * C + current_channel;
	// 		}
	// 	}
	// }
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
		this->input_gradients = gradients_ptr->back();

		Tensor<float, 3> *output = new Tensor<float, 3>(output_sizes);
		this->output_data = output;
		buffers_ptr->push_back(output);

		Tensor<float, 3> *output_gradient = new Tensor<float, 3>(output_data->dimensions());
		this->output_gradients = output_gradient;
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
		(*input_gradients).setZero();

		for (Index c = 0; c < output_sizes[0]; ++c) {
			for (Index i = 0; i < output_sizes[1]; ++i) {
				for (Index j = 0; j < output_sizes[2]; ++j) {

					for (Index x = 0; x < kernel_sizes[0]; ++x) {
						for (Index y = 0; y < kernel_sizes[1]; ++y) {
							Index in_i = i * stride + x;
							Index in_j = j * stride + y;

							if ((*output_data)(c, i, j) == (*input_data)(c, in_i, in_j)) {
								(*input_gradients)(c, in_i, in_j) = (*output_gradients)(c, i, j);
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

	MatrixXf weight_gradients;
	VectorXf bias_gradients;

	Index input_size;
	Index output_size;

	FullyConnectedLayer(vector<Tensor<float, 3> *> *buffers_ptr, vector<Tensor<float, 3> *> *gradients_ptr, Index output_dim) {
		weights = MatrixXf::Random(output_dim, buffers_ptr->back()->dimensions()[2]) / 1000.0f;
		biases = VectorXf::Random(output_dim) / 1000.0f;

		weight_gradients = MatrixXf::Random(output_dim, buffers_ptr->back()->dimensions()[2]) / 1000.0f;
		bias_gradients = VectorXf::Random(output_dim) / 1000.0f;

		this->input_size = buffers_ptr->back()->dimensions()[2];
		this->output_size = output_dim;

		attach_network(buffers_ptr, gradients_ptr);
	}

	void attach_network(vector<Tensor<float, 3> *> *buffers_ptr, vector<Tensor<float, 3> *> *gradients_ptr) override {
		this->input_data = buffers_ptr->back();
		this->input_gradients = gradients_ptr->back();
		input_gradients->setRandom();

		Tensor<float, 3> *output = new Tensor<float, 3>(1, 1, output_size);
		this->output_data = output;
		buffers_ptr->push_back(output);

		Tensor<float, 3> *output_gradient = new Tensor<float, 3>(output_data->dimensions());
		this->output_gradients = output_gradient;
		gradients_ptr->push_back(output_gradient);
	}

	void update_gradients() {
		// input gradient
		VectorXf output_gradients_vec(output_size);
		for (int i = 0; i < output_size; ++i) {
			output_gradients_vec(i) = (*output_gradients)(0, 0, i);
		}
		VectorXf input_gradient_vec = weights.transpose() * output_gradients_vec;

		float *in_grad_ptr = input_gradients->data();
		for (int i = 0; i < input_size; i++) {
			in_grad_ptr[i] = input_gradient_vec[i];
		}

		// weight gradients
		VectorXf input_data_vec(input_size);
		for (int i = 0; i < input_size; ++i) {
			input_data_vec(i) = (*input_data)(0, 0, i);
		}
		weight_gradients = output_gradients_vec * input_data_vec.transpose();

		// bias gradients
		bias_gradients = output_gradients_vec;
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

		cout << "Conv grad weights:" << endl;
		for (int j = 0; j < 10; j++) {
			cout << weight_gradients.data()[j] << " ";
		}

		cout << endl;
		cout << "Conv grad biases:" << endl;
		for (int j = 0; j < 10; j++) {
			cout << bias_gradients.data()[j] << " ";
		}
		cout << endl;

		weights -= learning_rate * weight_gradients;
		biases -= learning_rate * bias_gradients;
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
						(*input_gradients)(i, j, k) = 0;
					} else {
						(*input_gradients)(i, j, k) = (*output_gradients)(i, j, k);
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
					sum += (*output_gradients)(0, 0, j) * (*output_data)(0, 0, i) * (1 - (*output_data)(0, 0, i));
				} else {
					sum += -(*output_gradients)(0, 0, j) * (*output_data)(0, 0, i) * (*output_data)(0, 0, j);
				}
			}
			(*input_gradients)(0, 0, i) = sum;
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
		this->input_gradients = gradients_ptr->back();

		Tensor<float, 3> *output = new Tensor<float, 3>(output_sizes);
		this->output_data = output;
		buffers_ptr->push_back(output);

		Tensor<float, 3> *output_gradient = new Tensor<float, 3>(output_data->dimensions());
		this->output_gradients = output_gradient;
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
					(*input_gradients)(i, j, k) = (*output_gradients)(0, 0, flattten_index);
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

	void set_input_data(Tensor<float, 3> *input_image) {
		data_buffer[0] = input_image;
		layers[0]->input_data = input_image;
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

		// for (int i = 0; i < number_of_layers + 1; i++) {
		// 	Tensor<float, 3> tensor = *gradient_buffer[i];
		// 	float *tensor_ptr = tensor.data();
		//
		// 	const array<Index, 3> dims = tensor.dimensions();
		// 	float counter = 0;
		// 	for (int j = 0; j < 100 && j < dims[0] * dims[1] * dims[2]; j++) {
		// 		cout << tensor_ptr[j] << " ";
		// 		if (tensor_ptr[j] != 0.0f) {
		// 			counter++;
		// 		}
		// 	}
		// 	cout << endl;
		// 	cout << "Layer #" << i << ": " << counter / 100 << "\n";
		// }

		for (int i = number_of_layers - 1; i >= 0; i--) {
			layers[i]->backward(learning_rate);
		}
	}

	void print_buffers() {
		for (auto it = data_buffer.begin(); it != data_buffer.end(); it++) {
			cout << (*it) << endl;
			cout << "bruh" << endl;
		}
	}
};
