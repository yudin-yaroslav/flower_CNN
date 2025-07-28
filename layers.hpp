#include "dataset.hpp"
#include "tensor_print.hpp"
#include <iostream>
#include <memory>
#include <random>
#include <vector>

#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>
#define EIGEN_DONT_PARALLELIZE false

using namespace Eigen;
using std::cout, std::endl, std::cerr;
using std::unique_ptr, std::shared_ptr, std::make_unique, std::make_shared;
using std::vector;

inline Index get_plain_index(Index x, Index y, Index x_dim, Index y_dim) { return y * x_dim + x; }
inline Index get_plain_index(Index x, Index y, Index z, Index x_dim, Index y_dim, Index z_dim) {
	return z * x_dim * y_dim + y * x_dim + x;
}
inline Index get_plain_index(Index x, Index y, Index z, Index w, Index x_dim, Index y_dim, Index z_dim, Index w_dim) {
	return w * x_dim * y_dim * z_dim + z * x_dim * y_dim + y * x_dim + x;
}

class Layer {
  public:
	shared_ptr<Tensor<float, 4>> input_data = nullptr;
	shared_ptr<Tensor<float, 4>> output_data = nullptr;

	shared_ptr<Tensor<float, 4>> input_gradients = nullptr;
	shared_ptr<Tensor<float, 4>> output_gradients = nullptr;

	Index batch_size;

	virtual void attach_network(vector<shared_ptr<Tensor<float, 4>>> *buffers_ptr,
								vector<shared_ptr<Tensor<float, 4>>> *gradients_ptr) {
		this->input_data = buffers_ptr->back();
		this->input_gradients = gradients_ptr->back();

		this->output_data = make_shared<Tensor<float, 4>>(input_data->dimensions());
		buffers_ptr->push_back(output_data);

		this->output_gradients = make_shared<Tensor<float, 4>>(output_data->dimensions());
		gradients_ptr->push_back(output_gradients);
	};

	virtual void forward() = 0;
	virtual void backward(float learning_rate) = 0;

	virtual ~Layer() {
		input_data.reset();
		output_data.reset();
		input_gradients.reset();
		output_gradients.reset();
	};
};

class ConvolutionalLayer : public Layer {
  public:
	// BUG:
	MatrixXi input_reshaped_indices;

	MatrixXi out_grad_reshaped_for_input_indices;
	MatrixXi out_grad_reshaped_for_kernel_indices;

	VectorXi kernel_reshaped_indices;

	MatrixXf bias_gradients;
	MatrixXf kernel_gradients;

	MatrixXf kernel;
	MatrixXf biases;

	// input_sizes.shape = (batch_size, channels, height, width) = (B, C, H, W)
	array<Index, 4> input_sizes;
	array<Index, 4> output_sizes;
	array<Index, 2> kernel_sizes;

	Index stride;

	ConvolutionalLayer(vector<shared_ptr<Tensor<float, 4>>> *buffers_ptr, vector<shared_ptr<Tensor<float, 4>>> *gradients_ptr,
					   const array<Index, 2> &kernel_dims, Index filters, Index stride) {
		this->input_sizes = buffers_ptr->back()->dimensions();
		this->kernel_sizes = kernel_dims;
		this->stride = stride;

		this->K = filters;

		this->B = input_sizes[0];
		this->C = input_sizes[1];
		this->H = input_sizes[2];
		this->W = input_sizes[3];

		this->R = kernel_sizes[0];
		this->S = kernel_sizes[1];

		this->P = (H - R) / stride + 1;
		this->Q = (W - S) / stride + 1;

		if (P <= 0 || Q <= 0) {
			cerr << "The ouput size is too small\n";
		}

		this->H_back = (stride - 1) * (P - 1) + P + (R - 1) * 2;
		this->W_back = (stride - 1) * (Q - 1) + Q + (S - 1) * 2;
		this->P_back = H_back - R + 1;
		this->Q_back = W_back - S + 1;

		this->output_sizes = {B, K, P, Q};

		std::default_random_engine rng(std::random_device{}());
		float fan_in = R * S * C;
		float kaiming_bound = std::sqrt(6.0f / fan_in);

		std::uniform_real_distribution<float> weight_dist(-kaiming_bound, kaiming_bound);
		kernel = MatrixXf::NullaryExpr(K, R * S * C, [&]() { return weight_dist(rng); });

		float bias_bound = 1.0f / std::sqrt(fan_in);
		std::uniform_real_distribution<float> bias_dist(-bias_bound, bias_bound);
		biases = MatrixXf::NullaryExpr(K, P * Q, [&]() { return bias_dist(rng); });

		bias_gradients = MatrixXf::Zero(K, P * Q);
		kernel_gradients = MatrixXf::Zero(K, R * S * C);

		attach_network(buffers_ptr, gradients_ptr);

		reindex_input_data_for_forw();
		reindex_output_gradient_for_input();
		reindex_output_gradient_for_kernel();
		reindex_kernels();
	}

	void attach_network(vector<shared_ptr<Tensor<float, 4>>> *buffers_ptr,
						vector<shared_ptr<Tensor<float, 4>>> *gradients_ptr) override {
		this->input_data = buffers_ptr->back();
		this->input_gradients = gradients_ptr->back();

		this->output_data = make_shared<Tensor<float, 4>>(output_sizes);
		buffers_ptr->push_back(output_data);

		this->output_gradients = make_shared<Tensor<float, 4>>(output_data->dimensions());
		gradients_ptr->push_back(output_gradients);
	}

	void forward() override {
		for (Index b = 0; b < B; b++) {
			MatrixXf input_reshaped(C * R * S, P * Q);

			float *in_ptr = input_data->data();
			float *in_reshaped_ptr = input_reshaped.data();

			for (Index i = 0; i < (C * R * S) * (P * Q); ++i) {
				in_reshaped_ptr[i] = in_ptr[input_reshaped_indices(b, i)];
			}
			MatrixXf result = kernel * input_reshaped + biases; // shape = (K, PQ)

			for (Index f = 0; f < K; f++) {
				for (Index i = 0; i < P * Q; ++i) {
					(*output_data)(b, f, i / Q, i % Q) = result(f, i);
				}
			}
		}
	}

	void update_input_gradients() {
		MatrixXf kernel_reshaped(C, K * R * S);
		float *kernel_ptr = kernel.data();
		float *kernel_reshaped_ptr = kernel_reshaped.data();

		for (Index i = 0; i < (C * K * R * S); ++i) {
			kernel_reshaped_ptr[i] = kernel_ptr[kernel_reshaped_indices[i]];
		}

		for (Index b = 0; b < B; b++) {
			MatrixXf output_grad_reshaped_in(K * R * S, P_back * Q_back);
			float *out_ptr_in = output_gradients->data();
			float *out_reshaped_ptr_in = output_grad_reshaped_in.data();

			for (Index i = 0; i < (K * R * S) * (P_back * Q_back); ++i) {
				if (out_grad_reshaped_for_input_indices(b, i) == B * K * P * Q) {
					out_reshaped_ptr_in[i] = 0.0f;
				} else {
					out_reshaped_ptr_in[i] = out_ptr_in[out_grad_reshaped_for_input_indices(b, i)];
				}
			}

			MatrixXf input_gradient_mat = kernel_reshaped * output_grad_reshaped_in;

			for (Index channel = 0; channel < C; ++channel) {
				for (Index r = 0; r < H; ++r) {
					for (Index c = 0; c < W; ++c) {
						(*input_gradients)(b, channel, r, c) = input_gradient_mat(channel, r * W + c);
					}
				}
			}
		}
	}

	void update_bias_gradients() {
		bias_gradients.setZero();

		for (Index f = 0; f < K; f++) {
			for (Index b = 0; b < B; b++) {
				for (Index r = 0; r < P; r++) {
					for (Index c = 0; c < Q; c++) {
						bias_gradients(f, r + c * P) += (*output_gradients)(b, f, r, c);
					}
				}
			}
		}
	}

	void update_kernel_gradients() {
		kernel_gradients.setZero();

		for (Index b = 0; b < B; b++) {
			MatrixXf output_grad_reshaped_ker(K, P * Q);
			float *out_ptr_ker = output_gradients->data();
			float *out_reshaped_ptr_ker = output_grad_reshaped_ker.data();

			for (Index i = 0; i < K * P * Q; ++i) {
				out_reshaped_ptr_ker[i] = out_ptr_ker[out_grad_reshaped_for_kernel_indices(b, i)];
			}

			MatrixXf input_data_reshaped_ker(C * R * S, P * Q);
			float *in_ptr = input_data->data();
			float *in_reshaped_ptr = input_data_reshaped_ker.data();

			for (Index i = 0; i < C * R * S * P * Q; ++i) {
				in_reshaped_ptr[i] = in_ptr[input_reshaped_indices(b, i)];
			}

			kernel_gradients += output_grad_reshaped_ker * input_data_reshaped_ker.transpose();
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
	Index B, C, R, S, P, Q, H, W, K;
	Index H_back, W_back, P_back, Q_back;

	void reindex_input_data_for_forw() {
		input_reshaped_indices = MatrixXi(B, C * R * S * P * Q);

		for (Index b = 0; b < B; b++) {
			for (Index r = 0; r < C * R * S; r++) {
				for (Index c = 0; c < P * Q; c++) {

					Index current_channel = r / (R * S);

					Index kernel_r = c / Q;
					Index kernel_c = c % Q;

					Index value_r = (r % (R * S)) / R;
					Index value_c = (r % (R * S)) % R;

					Index pos_r = kernel_r * stride + value_r;
					Index pos_c = kernel_c * stride + value_c;

					input_reshaped_indices(b, c * S * R * C + r) = get_plain_index(b, current_channel, pos_r, pos_c, B, C, H, W);
				}
			}
		}
	}

	void reindex_output_gradient_for_input() {
		out_grad_reshaped_for_input_indices = MatrixXi::Constant(B, R * S * K * P_back * Q_back, B * K * P * Q);

		for (Index b = 0; b < B; b++) {
			for (Index r = 0; r < K * R * S; r++) {
				for (Index c = 0; c < P_back * Q_back; c++) {

					Index current_filter = r / (R * S);

					Index kernel_r = c / Q_back;
					Index kernel_c = c % Q_back;

					Index value_r = (r % (R * S)) / R;
					Index value_c = (r % (R * S)) % R;

					Index pos_r = kernel_r * stride + value_r;
					Index pos_c = kernel_c * stride + value_c;

					if (pos_r >= R - 1 && pos_r <= H + R - 2 && pos_c >= S - 1 && pos_c <= W + S - 2) {
						pos_r -= (R - 1);
						pos_c -= (S - 1);

						if (pos_r % stride == 0 && pos_c % stride == 0) {
							pos_r /= stride;
							pos_c /= stride;

							out_grad_reshaped_for_input_indices(b, c * S * R * K + r) =
								get_plain_index(b, current_filter, pos_r, pos_c, B, K, P, Q);
						}
					}
				}
			}
		}
	}

	void reindex_output_gradient_for_kernel() {
		out_grad_reshaped_for_kernel_indices = MatrixXi(B, K * P * Q);
		for (Index b = 0; b < B; b++) {
			for (Index r = 0; r < K; r++) {
				for (Index c = 0; c < P * Q; c++) {
					Index current_filter = r;

					Index pos_r = c / Q;
					Index pos_c = c % Q;

					out_grad_reshaped_for_kernel_indices(b, c * K + r) =
						get_plain_index(b, current_filter, pos_r, pos_c, B, K, P, Q);
				}
			}
		}
	}

	void reindex_kernels() {
		kernel_reshaped_indices.resize(C * R * S * K);
		for (int r = 0; r < C; r++) {
			for (int c = 0; c < K * R * S; c++) {
				Index current_channel = r;
				Index current_filter = c / (R * S);

				Index kernel_index_old = c % (R * S);
				Index kernel_index_new = R * S - 1 - kernel_index_old;

				kernel_reshaped_indices[c * C + r] = current_filter + K * kernel_index_new + K * R * S * current_channel;
			}
		}
	}
};

class MaxPoolingLayer : public Layer {
  public:
	array<Index, 4> input_sizes;
	array<Index, 2> kernel_sizes;
	array<Index, 4> output_sizes;

	Index stride;

	MaxPoolingLayer(vector<shared_ptr<Tensor<float, 4>>> *buffers_ptr, vector<shared_ptr<Tensor<float, 4>>> *gradients_ptr,
					const array<Index, 2> &kernel_dims, Index stride) {
		this->input_sizes = buffers_ptr->back()->dimensions();
		this->kernel_sizes = kernel_dims;
		this->stride = stride;

		this->output_sizes = {input_sizes[0], input_sizes[1], (input_sizes[2] - kernel_sizes[0]) / stride + 1,
							  (input_sizes[3] - kernel_sizes[1]) / stride + 1};

		if (output_sizes[1] <= 0 || output_sizes[2] <= 0) {
			cerr << "The ouput size is too small\n";
		}

		attach_network(buffers_ptr, gradients_ptr);
	}

	void attach_network(vector<shared_ptr<Tensor<float, 4>>> *buffers_ptr,
						vector<shared_ptr<Tensor<float, 4>>> *gradients_ptr) override {
		this->input_data = buffers_ptr->back();
		this->input_gradients = gradients_ptr->back();

		this->output_data = make_shared<Tensor<float, 4>>(output_sizes);
		buffers_ptr->push_back(output_data);

		this->output_gradients = make_shared<Tensor<float, 4>>(output_data->dimensions());
		gradients_ptr->push_back(output_gradients);
	}

	void forward() override {
		for (Index b = 0; b < output_sizes[0]; ++b) {
			for (Index c = 0; c < output_sizes[1]; ++c) {
				for (Index i = 0; i < output_sizes[2]; ++i) {
					for (Index j = 0; j < output_sizes[3]; ++j) {
						Index in_i = i * stride;
						Index in_j = j * stride;
						float max_val = (*input_data)(b, c, in_i, in_j);

						for (Index x = 0; x < kernel_sizes[0]; ++x) {
							for (Index y = 0; y < kernel_sizes[1]; ++y) {
								Index in_i = i * stride + x;
								Index in_j = j * stride + y;

								max_val = std::max(max_val, (*input_data)(b, c, in_i, in_j));
							}
						}
						(*output_data)(b, c, i, j) = max_val;
					}
				}
			}
		}
	}

	void backward(float learning_rate) override {
		(*input_gradients).setZero();

		for (Index b = 0; b < output_sizes[0]; ++b) {
			for (Index c = 0; c < output_sizes[1]; ++c) {
				for (Index i = 0; i < output_sizes[2]; ++i) {
					for (Index j = 0; j < output_sizes[3]; ++j) {

						for (Index x = 0; x < kernel_sizes[0]; ++x) {
							for (Index y = 0; y < kernel_sizes[1]; ++y) {
								Index in_i = i * stride + x;
								Index in_j = j * stride + y;

								if ((*output_data)(b, c, i, j) == (*input_data)(b, c, in_i, in_j)) {
									(*input_gradients)(b, c, in_i, in_j) = (*output_gradients)(b, c, i, j);
									goto break_label;
								}
							}
						}

					break_label:;
					}
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

	Index batch_size;
	Index input_size;
	Index output_size;

	std::default_random_engine rng;
	std::uniform_real_distribution<float> uniform;

	FullyConnectedLayer(vector<shared_ptr<Tensor<float, 4>>> *buffers_ptr, vector<shared_ptr<Tensor<float, 4>>> *gradients_ptr,
						Index output_dim) {
		this->batch_size = buffers_ptr->back()->dimensions()[0];
		this->input_size = buffers_ptr->back()->dimensions()[3];
		this->output_size = output_dim;

		float bound = std::sqrt(6.0f / input_size);
		uniform = std::uniform_real_distribution<float>(-bound, bound);

		weights = MatrixXf::NullaryExpr(output_size, input_size, [&]() { return uniform(rng); });
		biases = VectorXf::NullaryExpr(output_size, [&]() { return uniform(rng); });

		weight_gradients = MatrixXf::Zero(output_size, input_size);
		bias_gradients = VectorXf::Zero(output_size);

		attach_network(buffers_ptr, gradients_ptr);
	}

	void attach_network(vector<shared_ptr<Tensor<float, 4>>> *buffers_ptr,
						vector<shared_ptr<Tensor<float, 4>>> *gradients_ptr) override {
		this->input_data = buffers_ptr->back();
		this->input_gradients = gradients_ptr->back();

		this->output_data = make_shared<Tensor<float, 4>>(batch_size, 1, 1, output_size);
		buffers_ptr->push_back(output_data);

		this->output_gradients = make_shared<Tensor<float, 4>>(output_data->dimensions());
		gradients_ptr->push_back(output_gradients);
	}

	void forward() override {
		for (Index b = 0; b < batch_size; b++) {
			VectorXf in_vec(input_size);
			for (int i = 0; i < input_size; ++i) {
				in_vec(i) = (*input_data)(b, 0, 0, i);
			}

			VectorXf out_vec = weights * in_vec + biases;

			for (int i = 0; i < output_size; ++i) {
				(*output_data)(b, 0, 0, i) = out_vec(i);
			}
		}
	}

	void update_gradients() {
		weight_gradients.setZero();
		bias_gradients.setZero();

		for (Index b = 0; b < batch_size; b++) {
			// input gradient
			VectorXf output_gradients_vec(output_size);
			for (int i = 0; i < output_size; ++i) {
				output_gradients_vec(i) = (*output_gradients)(b, 0, 0, i);
			}
			VectorXf input_gradient_vec = weights.transpose() * output_gradients_vec;

			float *in_grad_ptr = input_gradients->data();
			for (int i = 0; i < input_size; i++) {
				in_grad_ptr[get_plain_index(b, 0, 0, i, batch_size, 1, 1, input_size)] = input_gradient_vec[i];
			}

			// weight gradients
			VectorXf input_data_vec(input_size);

			for (int i = 0; i < input_size; ++i) {
				input_data_vec(i) = (*input_data)(b, 0, 0, i);
			}
			weight_gradients += output_gradients_vec * input_data_vec.transpose();

			// bias gradients
			bias_gradients += output_gradients_vec;
		}
	}

	void backward(float learning_rate) override {
		update_gradients();

		weights -= learning_rate * weight_gradients;
		biases -= learning_rate * bias_gradients;
	}
};

class ReLULayer : public Layer {
  public:
	array<Index, 4> input_sizes;

	ReLULayer(vector<shared_ptr<Tensor<float, 4>>> *buffers_ptr, vector<shared_ptr<Tensor<float, 4>>> *gradients_ptr) {
		this->input_sizes = buffers_ptr->back()->dimensions();

		attach_network(buffers_ptr, gradients_ptr);
	};

	void forward() override {
		(*output_data) = (*input_data).unaryExpr([](float x) { return std::max(0.0f, x); });
	}

	void backward(float learning_rate) override {
		for (Index b = 0; b < input_sizes[0]; ++b) {
			for (Index c = 0; c < input_sizes[1]; ++c) {
				for (Index i = 0; i < input_sizes[2]; ++i) {
					for (Index j = 0; j < input_sizes[3]; ++j) {

						if ((*output_data)(b, c, i, j) == 0) {
							(*input_gradients)(b, c, i, j) = 0;
						} else {
							(*input_gradients)(b, c, i, j) = (*output_gradients)(b, c, i, j);
						}
					}
				}
			}
		}
	}
};

class LeakyReLULayer : public Layer {
  public:
	array<Index, 4> input_sizes;
	const float negative_slope = 0.01f;

	LeakyReLULayer(vector<shared_ptr<Tensor<float, 4>>> *buffers_ptr, vector<shared_ptr<Tensor<float, 4>>> *gradients_ptr) {
		this->input_sizes = buffers_ptr->back()->dimensions();
		attach_network(buffers_ptr, gradients_ptr);
	}

	void forward() override {
		(*output_data) = (*input_data).unaryExpr([this](float x) { return x > 0 ? x : negative_slope * x; });
	}

	void backward(float learning_rate) override {
		for (Index b = 0; b < input_sizes[0]; ++b) {
			for (Index c = 0; c < input_sizes[1]; ++c) {
				for (Index i = 0; i < input_sizes[2]; ++i) {
					for (Index j = 0; j < input_sizes[3]; ++j) {

						float x = (*input_data)(b, c, i, j);
						float grad = (*output_gradients)(b, c, i, j);
						(*input_gradients)(b, c, i, j) = (x > 0) ? grad : negative_slope * grad;
					}
				}
			}
		}
	}
};

class SoftMaxLayer : public Layer {
  public:
	Index layer_size;
	Index batch_size;

	SoftMaxLayer(vector<shared_ptr<Tensor<float, 4>>> *buffers_ptr, vector<shared_ptr<Tensor<float, 4>>> *gradients_ptr) {
		const array<Index, 4> dims = buffers_ptr->back()->dimensions();

		if (buffers_ptr->back()->dimensions()[1] != 1 || buffers_ptr->back()->dimensions()[2] != 1) {
			cerr << "SoftMax works EXCLUSIVELY after fully-connected layers, but the shape of input layer is (" +
						std::to_string(dims[0]) + ", " + std::to_string(dims[1]) + ", " + std::to_string(dims[2]) + ", " +
						std::to_string(dims[3]) + ")\n";
		}
		this->batch_size = dims[0];
		this->layer_size = dims[3];

		attach_network(buffers_ptr, gradients_ptr);
	};

	void forward() override {
		for (Index b = 0; b < batch_size; b++) {
			float max_value = (*input_data)(b, 0, 0, 0);

			for (Index i = 0; i < layer_size; ++i) {
				max_value = std::max(max_value, (*input_data)(b, 0, 0, i));
			}

			float sum = 0.0f;
			for (Index i = 0; i < layer_size; i++) {
				sum += exp((*input_data)(b, 0, 0, i) - max_value);
			}

			for (Index i = 0; i < layer_size; ++i) {
				(*output_data)(b, 0, 0, i) = exp((*input_data)(b, 0, 0, i) - max_value) / sum;
			}
		}
	}

	void backward(float learning_rate) override {
		for (Index b = 0; b < batch_size; b++) {
			for (Index i = 0; i < layer_size; i++) {
				float sum = 0.0f;
				for (Index j = 0; j < layer_size; j++) {
					if (j == i) {
						sum += (*output_gradients)(b, 0, 0, j) * (*output_data)(b, 0, 0, i) * (1 - (*output_data)(b, 0, 0, i));
					} else {
						sum += -(*output_gradients)(b, 0, 0, j) * (*output_data)(b, 0, 0, i) * (*output_data)(b, 0, 0, j);
					}
				}
				(*input_gradients)(b, 0, 0, i) = sum;
			}
		}
	}
};

class ReshapeLayer : public Layer {
  public:
	array<Index, 4> input_sizes;
	array<Index, 4> output_sizes;

	ReshapeLayer(vector<shared_ptr<Tensor<float, 4>>> *buffers_ptr, vector<shared_ptr<Tensor<float, 4>>> *gradients_ptr) {
		this->input_sizes = buffers_ptr->back()->dimensions();
		this->output_sizes = {input_sizes[0], 1, 1, input_sizes[1] * input_sizes[2] * input_sizes[3]};

		attach_network(buffers_ptr, gradients_ptr);
	}

	void attach_network(vector<shared_ptr<Tensor<float, 4>>> *buffers_ptr,
						vector<shared_ptr<Tensor<float, 4>>> *gradients_ptr) override {
		this->input_data = buffers_ptr->back();
		this->input_gradients = gradients_ptr->back();

		this->output_data = make_shared<Tensor<float, 4>>(output_sizes);
		buffers_ptr->push_back(output_data);

		this->output_gradients = make_shared<Tensor<float, 4>>(output_data->dimensions());
		gradients_ptr->push_back(output_gradients);
	}

	void forward() override {
		for (Index b = 0; b < input_sizes[0]; ++b) {
			for (Index i = 0; i < input_sizes[1]; ++i) {
				for (Index j = 0; j < input_sizes[2]; ++j) {
					for (Index k = 0; k < input_sizes[3]; ++k) {
						Index plain_index = get_plain_index(i, j, k, input_sizes[1], input_sizes[2], input_sizes[3]);
						(*output_data)(b, 0, 0, plain_index) = (*input_data)(b, i, j, k);
					}
				}
			}
		}
	}

	void backward(float learning_rate = 0.0f) override {
		for (Index b = 0; b < input_sizes[0]; ++b) {
			for (Index i = 0; i < input_sizes[1]; ++i) {
				for (Index j = 0; j < input_sizes[2]; ++j) {
					for (Index k = 0; k < input_sizes[3]; ++k) {
						Index plain_index = get_plain_index(i, j, k, input_sizes[1], input_sizes[2], input_sizes[3]);
						(*input_gradients)(b, i, j, k) = (*output_gradients)(b, 0, 0, plain_index);
					}
				}
			}
		}
	}
};

class DropoutLayer : public Layer {
  public:
	Tensor<float, 2> mask;

	Index layer_size;
	Index batch_size;

	float dropout_rate = 0.3f;

	std::default_random_engine rng;
	std::bernoulli_distribution bernoulli;

	DropoutLayer(vector<shared_ptr<Tensor<float, 4>>> *buffers_ptr, vector<shared_ptr<Tensor<float, 4>>> *gradients_ptr,
				 float rate = 0.3f) {

		this->dropout_rate = rate;
		this->bernoulli = std::bernoulli_distribution(1.0 - rate);

		this->batch_size = buffers_ptr->back()->dimensions()[0];
		this->layer_size = buffers_ptr->back()->dimensions()[3];

		attach_network(buffers_ptr, gradients_ptr);

		mask = Tensor<float, 2>(batch_size, layer_size);
	}

	void forward() override {
		for (Index b = 0; b < batch_size; ++b) {
			for (Index l = 0; l < layer_size; ++l) {
				mask(b, l) = bernoulli(rng) ? 1.0f / (1.0f - dropout_rate) : 0.0f;
				(*output_data)(b, 0, 0, l) = (*input_data)(b, 0, 0, l) * mask(b, l);
			}
		}
	}

	void backward(float learning_rate) override {
		for (Index b = 0; b < batch_size; ++b) {
			for (Index l = 0; l < layer_size; ++l) {
				(*input_gradients)(b, 0, 0, l) = (*output_gradients)(b, 0, 0, l) * mask(b, l);
			}
		}
	}
};

class CNN {
  public:
	vector<shared_ptr<Tensor<float, 4>>> data_buffer;
	vector<shared_ptr<Tensor<float, 4>>> gradient_buffer;

	vector<unique_ptr<Layer>> layers;
	int number_of_layers;

	float learning_rate;
	int number_of_predictions;

	MatrixXf predictions;
	Index batch_size;

	CNN(const Tensor<float, 4> &image_data, float learning_rate, Index batch_size) {
		srand((unsigned int)time(0));

		data_buffer.push_back(make_shared<Tensor<float, 4>>(std::move(image_data)));
		gradient_buffer.push_back(make_shared<Tensor<float, 4>>(image_data.dimensions()));

		this->learning_rate = learning_rate;
		this->batch_size = batch_size;
		number_of_layers = 0;
	}

	CNN(const array<Index, 4> image_dims, float learning_rate, Index batch_size) {
		srand((unsigned int)time(0));

		data_buffer.push_back(make_shared<Tensor<float, 4>>(image_dims));
		gradient_buffer.push_back(make_shared<Tensor<float, 4>>(image_dims));

		this->learning_rate = learning_rate;
		this->batch_size = batch_size;
		number_of_layers = 0;
	}

	void set_input_data(const Tensor<float, 4> &image_data) {
		*data_buffer[0] = image_data;
		*layers[0]->input_data = image_data;
	}

	void add_convolutional_layer(const array<Index, 2> &kernel_dims, Index filters, Index stride) {
		layers.push_back(make_unique<ConvolutionalLayer>(&data_buffer, &gradient_buffer, kernel_dims, filters, stride));
		number_of_layers++;
	}

	void add_maxpooling_layer(const array<Index, 2> &kernel_dims, Index stride) {
		layers.push_back(make_unique<MaxPoolingLayer>(&data_buffer, &gradient_buffer, kernel_dims, stride));
		number_of_layers++;
	}

	void add_fully_connected_layer(Index output_dim) {
		layers.push_back(make_unique<FullyConnectedLayer>(&data_buffer, &gradient_buffer, output_dim));
		number_of_layers++;
	}

	void add_relu_layer() {
		layers.push_back(make_unique<ReLULayer>(&data_buffer, &gradient_buffer));
		number_of_layers++;
	}

	void add_leaky_relu_layer() {
		layers.push_back(make_unique<LeakyReLULayer>(&data_buffer, &gradient_buffer));
		number_of_layers++;
	}

	void add_flatten_layer() {
		layers.push_back(make_unique<ReshapeLayer>(&data_buffer, &gradient_buffer));
		number_of_layers++;
	}

	void add_dropout_layer(float dropout_rate) {
		layers.push_back(make_unique<DropoutLayer>(&data_buffer, &gradient_buffer, dropout_rate));
		number_of_layers++;
	}

	void add_softmax_layer() {
		unique_ptr<SoftMaxLayer> layer = make_unique<SoftMaxLayer>(&data_buffer, &gradient_buffer);

		this->number_of_predictions = layer->layer_size;
		this->predictions = MatrixXf(batch_size, number_of_predictions);

		layers.push_back(std::move(layer));
		number_of_layers++;
	}

	void forward() {
		for (int i = 0; i < number_of_layers; i++) {
			layers[i]->forward();
		}

		if (dynamic_cast<SoftMaxLayer *>(layers.back().get()) != nullptr) {
			for (int b = 0; b < batch_size; b++) {
				for (int i = 0; i < number_of_predictions; i++) {
					predictions(b, i) = (*data_buffer.back())(b, 0, 0, i);
				}
			}
		}
	}

	void backward(vector<int> correct_prediction) {
		if (dynamic_cast<SoftMaxLayer *>(layers.back().get()) != nullptr) {
			for (int b = 0; b < batch_size; b++) {
				for (int i = 0; i < number_of_predictions; i++) {
					(*gradient_buffer[number_of_layers])(b, 0, 0, i) =
						predictions(b, i) - (i == correct_prediction[b] ? 1.0f : 0.0f);
				}
			}
		}

		// cout << endl << endl;
		// for (int i = 2; i < number_of_layers + 1; i++) {
		// 	Tensor<float, 3> tensor = *gradient_buffer[i];
		// 	float *tensor_ptr = tensor.data();
		//
		// 	const array<Index, 3> dims = tensor.dimensions();
		// 	float counter = 0;
		// 	for (int j = 0; j < dims[0] * dims[1] * dims[2]; j++) {
		// 		if (tensor_ptr[j] != 0.0f) {
		// 			counter++;
		// 		}
		// 	}
		// 	for (int j = 0; j < 100 && j < dims[0] * dims[1] * dims[2]; j++) {
		// 		cout << tensor_ptr[j] << " ";
		// 	}
		// 	cout << endl;
		// 	cout << "Layer #" << i << ": " << counter << ", " << (dims[0] * dims[1] * dims[2]) << "\n";
		// }
		// cout << endl << endl;

		for (int i = number_of_layers - 1; i >= 0; i--) {
			layers[i]->backward(learning_rate);
		}
	}
};
