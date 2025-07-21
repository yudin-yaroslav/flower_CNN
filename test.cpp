#include "layers.hpp"
#include <bits/stdc++.h>
#include <chrono>
#include <utility>

using namespace std::chrono;
using std::pair;

vector<pair<string, bool>> tests;

void test_func(std::function<pair<string, bool>()> func) {
	auto t1 = high_resolution_clock::now();
	pair<string, bool> result = func();
	auto t2 = high_resolution_clock::now();

	auto ms_int = duration_cast<milliseconds>(t2 - t1);
	cout << endl << "Time duration:" << endl;
	cout << ms_int.count() << "ms\n";

	if (ms_int.count() > 1500) {
		result.second = false;
	}
	tests.push_back(result);

	if (result.second == true) {
		cout << "\n\033[1;32mTest passed ✔ \033[0m\n\n";
	} else {
		cout << "\n\033[1;31mTest failed ✘ \033[0m\n\n";
	}
}

void test_stat() {
	cout << "\n\033[1;33m===== Completed Tests =====\033[0m\n";

	int passed = 0, failed = 0;
	size_t max_len = 0;
	for (const auto &[name, _] : tests) {
		max_len = std::max(max_len, name.length());
	}

	for (const auto &[name, status] : tests) {
		cout << "  Test ";
		cout << std::left << std::setw(max_len) << name << ":  ";

		if (status) {
			cout << "\033[1;32m✔ PASSED\033[0m\n";
			++passed;
		} else {
			cout << "\033[1;31m✘ FAILED\033[0m\n";
			++failed;
		}
	}

	cout << "\n\033[1;34mSummary:\033[0m "
		 << "\033[1;32m" << passed << " passed\033[0m, "
		 << "\033[1;31m" << failed << " failed\033[0m.\n";
}

pair<string, bool> test_conv_forward_filters() {
	const string name = "Convolutional Forward Value (filters)";
	cout << "\033[1;33m===== Test " + name + " =====\033[0m\n";

	Tensor<float, 3> *input = new Tensor<float, 3>(1, 3, 3);

	float val = 0;
	for (Index i = 0; i < 1; ++i)
		for (Index j = 0; j < 3; ++j)
			for (Index k = 0; k < 3; ++k)
				(*input)(i, j, k) = val++;

	cout << "Input:" << endl;
	// cout << (*input) << endl << endl;
	PrintTensorNumpy(*input, 1, 3, 3);

	CNN net(input, 0.1);

	net.add_convolutional_layer(array<Index, 2>{2, 2}, 2, 1);

	auto layer_ptr = dynamic_cast<ConvolutionalLayer *>(net.layers[0]);

	cout << endl << "Kernel:" << endl;
	for (Index f = 0; f < layer_ptr->filters; ++f) {
		for (Index c = 0; c < layer_ptr->channels; ++c) {
			for (Index x = 0; x < layer_ptr->kernel_sizes[0]; ++x)
				for (Index y = 0; y < layer_ptr->kernel_sizes[1]; ++y)
					layer_ptr->kernel(f, c * layer_ptr->kernel_sizes[0] * layer_ptr->kernel_sizes[1] +
											 x * layer_ptr->kernel_sizes[1] + y) = 100 * f + 10 * c + x + 0.1 * y;
		}
	}
	PrintMatrixNumpy(layer_ptr->kernel, layer_ptr->filters,
					 layer_ptr->channels * layer_ptr->kernel_sizes[0] * layer_ptr->kernel_sizes[1]);

	cout << endl << endl << "Biases:" << endl;
	for (Index f = 0; f < layer_ptr->filters; ++f) {
		for (Index x = 0; x < layer_ptr->output_sizes[1]; ++x)
			for (Index y = 0; y < layer_ptr->output_sizes[2]; ++y)
				layer_ptr->biases(f, x * layer_ptr->output_sizes[2] + y) = f + x + y;
	}
	PrintMatrixNumpy(layer_ptr->biases, layer_ptr->filters, layer_ptr->output_sizes[0] * layer_ptr->output_sizes[1]);
	cout << endl << endl;

	net.forward();

	Tensor<float, 3> output_correct(2, 2, 2);
	output_correct(0, 0, 0) = 7.5;
	output_correct(0, 0, 1) = 10.7;
	output_correct(0, 1, 0) = 15.1;
	output_correct(0, 1, 1) = 18.3;
	output_correct(1, 0, 0) = 808.5;
	output_correct(1, 0, 1) = 1211.7;
	output_correct(1, 1, 0) = 2016.1;
	output_correct(1, 1, 1) = 2419.3;

	Tensor<float, 3> output_computed = *net.data_buffer.back();

	cout << endl << endl << "Output:" << endl;
	// cout << (*net.buffers.back()) << endl;
	PrintTensorNumpy(output_computed, layer_ptr->output_sizes[0], layer_ptr->output_sizes[1], layer_ptr->output_sizes[2]);

	for (Index i = 0; i < 2; ++i) {
		for (Index j = 0; j < 2; ++j) {
			for (Index k = 0; k < 2; ++k) {
				if (abs(output_correct(i, j, k) - output_computed(i, j, k)) > 1e-3) {
					return {name, false};
				}
			}
		}
	}

	return {name, true};
}

pair<string, bool> test_conv_forward_channels() {
	const string name = "Convolutional Forward Value (channels)";
	cout << "\033[1;33m===== Test " + name + " =====\033[0m\n";

	Tensor<float, 3> *input = new Tensor<float, 3>(2, 3, 3);

	float val = 0;
	for (Index i = 0; i < 2; ++i)
		for (Index j = 0; j < 3; ++j)
			for (Index k = 0; k < 3; ++k)
				(*input)(i, j, k) = val++;

	cout << "Input:" << endl;
	// cout << (*input) << endl << endl;
	PrintTensorNumpy(*input, 2, 3, 3);

	CNN net(input, 0.1);

	net.add_convolutional_layer(array<Index, 2>{2, 2}, 1, 1);

	auto layer_ptr = dynamic_cast<ConvolutionalLayer *>(net.layers[0]);

	cout << endl << "Kernel:" << endl;
	for (Index f = 0; f < layer_ptr->filters; ++f) {
		for (Index c = 0; c < layer_ptr->channels; ++c) {
			for (Index x = 0; x < layer_ptr->kernel_sizes[0]; ++x)
				for (Index y = 0; y < layer_ptr->kernel_sizes[1]; ++y)
					layer_ptr->kernel(f, c * layer_ptr->kernel_sizes[0] * layer_ptr->kernel_sizes[1] +
											 x * layer_ptr->kernel_sizes[1] + y) = 100 * f + 10 * c + x + 0.1 * y;
		}
	}
	PrintMatrixNumpy(layer_ptr->kernel, layer_ptr->filters,
					 layer_ptr->channels * layer_ptr->kernel_sizes[0] * layer_ptr->kernel_sizes[1]);

	cout << endl << endl << "Biases:" << endl;
	for (Index f = 0; f < layer_ptr->filters; ++f) {
		for (Index x = 0; x < layer_ptr->output_sizes[1]; ++x)
			for (Index y = 0; y < layer_ptr->output_sizes[2]; ++y)
				layer_ptr->biases(f, x * layer_ptr->output_sizes[2] + y) = f + x + y;
	}
	PrintMatrixNumpy(layer_ptr->biases, layer_ptr->filters, layer_ptr->output_sizes[0] * layer_ptr->output_sizes[1]);
	cout << endl << endl;

	net.forward();

	Tensor<float, 3> output_correct(1, 2, 2);
	output_correct(0, 0, 0) = 474.8;
	output_correct(0, 0, 1) = 520.2;
	output_correct(0, 1, 0) = 609.0;
	output_correct(0, 1, 1) = 654.4;

	Tensor<float, 3> output_computed = *net.data_buffer.back();

	cout << endl << endl << "Output:" << endl;
	// cout << (*net.buffers.back()) << endl;
	PrintTensorNumpy(output_computed, layer_ptr->output_sizes[0], layer_ptr->output_sizes[1], layer_ptr->output_sizes[2]);

	for (Index i = 0; i < 2; ++i) {
		for (Index j = 0; j < 2; ++j) {
			for (Index k = 0; k < 2; ++k) {
				if (abs(output_correct(i, j, k) - output_computed(i, j, k)) > 1e-3) {
					return {name, false};
				}
			}
		}
	}

	return {name, true};
}

pair<string, bool> test_conv_forward_stride() {
	const string name = "Convolutional Forward Value (stride)";
	cout << "\033[1;33m===== Test " + name + " =====\033[0m\n";

	Tensor<float, 3> *input = new Tensor<float, 3>(1, 4, 4);

	float val = 0;
	for (Index c = 0; c < 1; ++c)
		for (Index i = 0; i < 4; ++i)
			for (Index j = 0; j < 4; ++j)
				(*input)(c, i, j) = val++;

	cout << "Input:" << endl;
	PrintTensorNumpy(*input, 1, 4, 4);

	CNN net(input, 0.1);

	net.add_convolutional_layer(array<Index, 2>{2, 2}, 1, 2);

	auto layer_ptr = dynamic_cast<ConvolutionalLayer *>(net.layers[0]);

	cout << endl << "Kernel:" << endl;
	for (Index f = 0; f < layer_ptr->filters; ++f) {
		for (Index c = 0; c < layer_ptr->channels; ++c) {
			for (Index x = 0; x < layer_ptr->kernel_sizes[0]; ++x)
				for (Index y = 0; y < layer_ptr->kernel_sizes[1]; ++y)
					layer_ptr->kernel(f, c * layer_ptr->kernel_sizes[0] * layer_ptr->kernel_sizes[1] +
											 x * layer_ptr->kernel_sizes[1] + y) = 1;
		}
	}
	PrintMatrixNumpy(layer_ptr->kernel, 1, 4);

	cout << endl << "Biases:" << endl;
	for (Index f = 0; f < 1; ++f)
		for (Index i = 0; i < 2; ++i)
			for (Index j = 0; j < 2; ++j)
				layer_ptr->biases(f, i * 2 + j) = 0;
	PrintMatrixNumpy(layer_ptr->biases, 1, 4);

	net.forward();

	Tensor<float, 3> output_correct(1, 2, 2);
	output_correct(0, 0, 0) = 0 + 1 + 4 + 5;
	output_correct(0, 0, 1) = 2 + 3 + 6 + 7;
	output_correct(0, 1, 0) = 8 + 9 + 12 + 13;
	output_correct(0, 1, 1) = 10 + 11 + 14 + 15;

	Tensor<float, 3> output_computed = *net.data_buffer.back();

	cout << endl << "Output:" << endl;
	PrintTensorNumpy(output_computed, 1, 2, 2);

	for (Index i = 0; i < 2; ++i) {
		for (Index j = 0; j < 2; ++j) {
			for (Index k = 0; k < 1; ++k) {
				if (abs(output_correct(k, i, j) - output_computed(k, i, j)) > 1e-3) {
					return {name, false};
				}
			}
		}
	}

	return {name, true};
}

pair<string, bool> test_conv_forward_speed() {
	const string name = "Convolutional Forward Speed";
	cout << "\033[1;33m===== Test " + name + " =====\033[0m\n";

	Tensor<float, 3> *input = new Tensor<float, 3>(3, 256, 256);

	int val = 1;
	for (Index i = 0; i < 3; ++i)
		for (Index j = 0; j < 256; ++j)
			for (Index k = 0; k < 256; ++k)
				(*input)(i, j, k) = static_cast<float>((val++) % 16) / 16;

	CNN net(input, 0.1);

	net.add_convolutional_layer(array<Index, 2>{12, 12}, 96, 4);
	net.forward();

	return {name, true};
}

pair<string, bool> test_fully_connected_forward_value() {
	const string name = "Fully Connected Forward Value";
	cout << "\033[1;33m===== Test " + name + " =====\033[0m\n";

	Tensor<float, 3> *input = new Tensor<float, 3>(1, 1, 4);
	for (int i = 0; i < 4; ++i) {
		(*input)(0, 0, i) = i + 1;
	}

	cout << "Input:\n";
	cout << *input << endl;

	CNN net(input, 0.1);
	net.add_fully_connected_layer(3);

	auto layer_ptr = dynamic_cast<FullyConnectedLayer *>(net.layers[0]);

	layer_ptr->weights << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12;

	cout << "\nWeights:\n";
	cout << layer_ptr->weights << endl;

	net.forward();

	Tensor<float, 3> output_correct(1, 1, 3);
	output_correct(0, 0, 0) = 30;
	output_correct(0, 0, 1) = 70;
	output_correct(0, 0, 2) = 110;

	Tensor<float, 3> &output_computed = *net.data_buffer.back();

	cout << "\nOutput:\n";
	cout << output_computed << endl;

	for (int i = 0; i < 3; ++i) {
		if (output_correct(0, 0, i) != output_computed(0, 0, i)) {
			return {name, false};
		}
	}

	return {name, true};
}

pair<string, bool> test_fully_connected_forward_speed() {
	const string name = "Fully Connected Forward Speed";
	cout << "\033[1;33m===== Test " + name + " =====\033[0m\n";

	Tensor<float, 3> *input = new Tensor<float, 3>(1, 1, 1024);
	for (int i = 0; i < 4; ++i) {
		(*input)(0, 0, i) = i + 1;
	}

	CNN net(input, 0.1);
	net.add_fully_connected_layer(1024);

	net.forward();

	return {name, true};
}

pair<string, bool> test_maxpool_forward_value() {
	const string name = "Max Pooling Forward Value";
	cout << "\033[1;33m===== Test " + name + " =====\033[0m\n";

	Tensor<float, 3> *input = new Tensor<float, 3>(1, 4, 4);

	float val = 1;
	for (int i = 0; i < 4; ++i)
		for (int j = 0; j < 4; ++j)
			(*input)(0, i, j) = val++;

	cout << "\nInput:\n";
	cout << (*input) << endl;

	CNN net(input, 0.1);

	net.add_maxpooling_layer(array<Index, 2>{2, 2}, 2);

	net.forward();

	Tensor<float, 3> output_computed = *net.data_buffer.back();

	Tensor<float, 3> output_correct(1, 2, 2);
	output_correct(0, 0, 0) = 6;
	output_correct(0, 0, 1) = 8;
	output_correct(0, 1, 0) = 14;
	output_correct(0, 1, 1) = 16;

	cout << "\nOutput:\n";
	cout << output_computed << endl;

	for (int i = 0; i < 2; ++i)
		for (int j = 0; j < 2; ++j)
			if (output_computed(0, i, j) != output_correct(0, i, j))
				return {name, false};

	return {name, true};
}

pair<string, bool> test_maxpool_forward_speed() {
	const string name = "Max Pooling Forward Speed";
	cout << "\033[1;33m===== Test " + name + " =====\033[0m\n";

	Tensor<float, 3> *input = new Tensor<float, 3>(3, 256, 256);

	for (int c = 0; c < 3; ++c)
		for (int i = 0; i < 256; ++i)
			for (int j = 0; j < 256; ++j)
				(*input)(c, i, j) = rand() % 100;

	CNN net(input, 0.1);

	net.add_maxpooling_layer(array<Index, 2>{2, 2}, 2);

	net.forward();

	return {name, true};
}

pair<string, bool> test_relu_forward_value() {
	const string name = "Test ReLU Forward Value";
	cout << "\033[1;33m===== Test " + name + " =====\033[0m\n";

	Tensor<float, 3> *input = new Tensor<float, 3>(2, 2, 2);

	float val = 1;
	for (int i = 0; i < 2; ++i)
		for (int j = 0; j < 2; ++j)
			for (int k = 0; k < 2; ++k)
				(*input)(i, j, k) = -4 + val++;

	cout << "\nInput:\n";
	cout << (*input) << endl;

	CNN net(input, 0.1);

	net.add_relu_layer();

	net.forward();

	Tensor<float, 3> output_computed = *net.data_buffer.back();

	Tensor<float, 3> output_correct(2, 2, 2);
	output_correct(0, 0, 0) = 0;
	output_correct(0, 0, 1) = 0;
	output_correct(0, 1, 0) = 0;
	output_correct(0, 1, 1) = 0;
	output_correct(1, 0, 0) = 1;
	output_correct(1, 0, 1) = 2;
	output_correct(1, 1, 0) = 3;
	output_correct(1, 1, 1) = 4;

	cout << "\nOutput:\n";
	cout << output_computed << endl;

	for (int i = 0; i < 2; ++i)
		for (int j = 0; j < 2; ++j)
			for (int k = 0; k < 2; ++k)
				if (output_computed(i, j, k) != output_correct(i, j, k))
					return {name, false};

	return {name, true};
}

pair<string, bool> test_softmax_forward_value() {
	const string name = "Test SoftMax Forward Value";
	cout << "\033[1;33m===== Test " + name + " =====\033[0m\n";

	Tensor<float, 3> *input = new Tensor<float, 3>(1, 1, 4);
	for (int i = 0; i < 4; ++i) {
		(*input)(0, 0, i) = i + 1;
	}

	cout << "Input:\n";
	cout << *input << endl;

	CNN net(input, 0.1);
	net.add_softmax_layer();

	net.forward();

	Tensor<float, 3> output_correct(1, 1, 4);
	output_correct(0, 0, 0) = 0.0320;
	output_correct(0, 0, 1) = 0.0871;
	output_correct(0, 0, 2) = 0.2369;
	output_correct(0, 0, 3) = 0.6439;

	Tensor<float, 3> &output_computed = *net.data_buffer.back();

	cout << "\nOutput:\n";
	cout << output_computed << endl;

	for (int i = 0; i < 4; ++i) {
		if (fabs(output_correct(0, 0, i) - output_computed(0, 0, i)) > 1e-4) {
			return {name, false};
		}
	}

	return {name, true};
}

pair<string, bool> test_reshape_forward_value() {
	const string name = "Reshape Forward Value";
	cout << "\033[1;33m===== Test " + name + " =====\033[0m\n";

	Tensor<float, 3> *input = new Tensor<float, 3>(2, 2, 2);

	float val = 1;
	for (int i = 0; i < 2; ++i)
		for (int j = 0; j < 2; ++j)
			for (int k = 0; k < 2; ++k)
				(*input)(i, j, k) = val++;

	cout << "Input:" << endl;
	cout << (*input) << endl;

	CNN net(input, 0.1);
	net.add_flatten_layer();

	net.forward();

	Tensor<float, 3> output_correct(1, 1, 8);
	for (int i = 0; i < 8; i++) {
		output_correct(0, 0, i) = i + 1;
	}

	Tensor<float, 3> &output_computed = *net.data_buffer.back();

	cout << "\nOutput:\n";
	cout << output_computed << endl;

	for (int i = 0; i < 8; ++i) {
		if (output_correct(0, 0, i) != output_computed(0, 0, i)) {
			return {name, false};
		}
	}

	return {name, true};
}

pair<string, bool> test_cnn_forward() {
	const string name = "Test CNN Forward";
	cout << "\033[1;33m===== Test " + name + " =====\033[0m\n";

	Tensor<float, 3> *input = new Tensor<float, 3>(3, 256, 256);

	int val = 1;
	for (Index i = 0; i < 3; ++i)
		for (Index j = 0; j < 256; ++j)
			for (Index k = 0; k < 256; ++k)
				(*input)(i, j, k) = static_cast<float>((val++) % 16) / 16;

	CNN net(input, 0.1);

	net.add_convolutional_layer({12, 12}, 96, 1);
	net.add_relu_layer();

	net.add_maxpooling_layer({3, 3}, 2);

	net.add_convolutional_layer({5, 5}, 256, 1);
	net.add_relu_layer();
	net.add_convolutional_layer({3, 3}, 256, 1);
	net.add_relu_layer();
	net.add_convolutional_layer({3, 3}, 96, 1);
	net.add_relu_layer();

	net.add_maxpooling_layer({3, 3}, 2);

	net.add_flatten_layer();
	net.add_fully_connected_layer(512);
	net.add_fully_connected_layer(216);
	net.add_fully_connected_layer(102);
	net.add_softmax_layer();

	cout << "First 10 input values: ";
	for (Index j = 0; j < 10; ++j) {
		cout << (*net.data_buffer[0])(0, 0, j) << " ";
	}
	cout << "\n\n";

	cout << "Starting forward propagation..." << endl;

	auto t_start = high_resolution_clock::now();

	for (size_t i = 0; i < net.layers.size(); ++i) {
		auto &layer = net.layers[i];
		auto t0 = high_resolution_clock::now();

		layer->forward();

		auto t1 = high_resolution_clock::now();
		auto elapsed = duration_cast<milliseconds>(t1 - t0).count();

		auto output = net.data_buffer[i + 1];
		auto dims = output->dimensions();

		cout << "\n\033[1;34m-- Layer " << i << "; Time elapsed: " << elapsed << " ms; Output shape: (" << dims[0] << ", "
			 << dims[1] << ", " << dims[2] << ")\033[0m\n";

		cout << "First 10 values: ";
		for (Index j = 0; j < 10 && j < dims[2]; ++j) {
			cout << (*output)(0, 0, j) << " ";
		}
		cout << endl;
	}

	cout << "\n\033[1;34mResults bigger than 0.01: \033[0m\n";

	float sum = 0.0;
	auto result = net.data_buffer.back();
	for (Index j = 0; j < 102; ++j) {
		float value = (*result)(0, 0, j);
		if (value > 0.01) {
			cout << "result[" << j << "] = " << value << endl;
		}
		sum += value;
	}

	cout << "\nSum: " << sum << endl;

	return {name, true};
}

pair<string, bool> test_loss_to_softmax_gradient() {
	const string name = "Loss Gradient";
	cout << "\033[1;33m===== Test " + name + " =====\033[0m\n";

	Tensor<float, 3> *input = new Tensor<float, 3>(1, 1, 4);
	(*input)(0, 0, 0) = 0;
	(*input)(0, 0, 1) = -2;
	(*input)(0, 0, 2) = 3;
	(*input)(0, 0, 3) = 2;

	cout << "Input:\n";
	PrintTensorNumpy(*input, 1, 1, 4);

	CNN net(input, 0.1);
	net.add_softmax_layer();

	net.forward();

	Tensor<float, 3> &output_computed = *net.data_buffer.back();

	cout << "\nOutput:\n";
	PrintTensorNumpy(output_computed, 1, 1, 4);

	cout << "\n\033[1;34mStarting back propagation...\033[0m\n";

	net.backward(2);

	Tensor<float, 3> gradient_correct(1, 1, 4);
	gradient_correct(0, 0, 0) = 0.0174765;
	gradient_correct(0, 0, 1) = 0.00236518;
	gradient_correct(0, 0, 2) = -0.148976;
	gradient_correct(0, 0, 3) = 0.129134;
	Tensor<float, 3> gradient_computed = *net.gradient_buffer[1];

	cout << "\nComputed gradient:\n";
	PrintTensorNumpy(gradient_computed, 1, 1, 4);
	cout << endl;

	for (Index i = 0; i < 4; i++) {
		if (abs(gradient_computed(0, 0, i) - gradient_correct(0, 0, i)) > 1e-4) {
			return {name, false};
		}
	}

	return {name, true};
}

pair<string, bool> test_softmax_gradient() {
	const string name = "SoftMax Gradient";
	cout << "\033[1;33m===== Test " + name + " =====\033[0m\n";

	Tensor<float, 3> *input = new Tensor<float, 3>(1, 1, 4);
	for (int i = 0; i < 4; ++i) {
		(*input)(0, 0, i) = i - 2;
	}

	cout << "Input:\n";
	PrintTensorNumpy(*input, 1, 1, 4);

	CNN net(input, 0.1);
	net.add_softmax_layer();

	net.forward();

	Tensor<float, 3> &output_computed = *net.data_buffer.back();

	cout << "\nOutput:\n";
	PrintTensorNumpy(output_computed, 1, 1, 4);

	cout << "\n\033[1;34mStarting back propagation...\033[0m\n";

	net.backward(2);

	Tensor<float, 3> gradient_correct(1, 1, 4);
	gradient_correct(0, 0, 0) = -0.00337287;
	gradient_correct(0, 0, 1) = -0.00676819;
	gradient_correct(0, 0, 2) = -0.11910423;
	gradient_correct(0, 0, 3) = +0.12924531;
	Tensor<float, 3> gradient_new = *net.gradient_buffer[0];
	Tensor<float, 3> gradient_old = *net.gradient_buffer[1];

	cout << "\nCorrect gradient:\n";
	PrintTensorNumpy(gradient_correct, 1, 1, 4);

	cout << "\nOld gradient:\n";
	PrintTensorNumpy(gradient_old, 1, 1, 4);

	cout << "\nNew gradient:\n";
	PrintTensorNumpy(gradient_new, 1, 1, 4);

	for (Index i = 0; i < 4; i++) {
		if (abs(gradient_new(0, 0, i) - gradient_correct(0, 0, i)) > 1e-4) {
			return {name, false};
		}
	}

	return {name, true};
}

pair<string, bool> test_reshape_gradient() {
	const string name = "Reshape Gradient";
	cout << "\033[1;33m===== Test " + name + " =====\033[0m\n";

	Tensor<float, 3> *input = new Tensor<float, 3>(2, 2, 2);

	float val = 2.0f;
	for (int i = 0; i < 2; ++i)
		for (int j = 0; j < 2; ++j)
			for (int k = 0; k < 2; ++k)
				(*input)(i, j, k) = val++;

	cout << "Input:" << endl;
	PrintTensorNumpy(*input, 2, 2, 2);

	CNN net(input, 0.1);
	net.add_flatten_layer();
	net.add_softmax_layer();

	net.forward();

	Tensor<float, 3> &output_computed = *net.data_buffer.back();

	cout << "\nOutput:\n";
	PrintTensorNumpy(output_computed, 1, 1, 8);

	cout << "\n\033[1;34mStarting back propagation...\033[0m\n";

	net.backward(3);

	Tensor<float, 3> gradient_correct(2, 2, 2);
	gradient_correct(0, 0, 0) = -6.49077e-05;
	gradient_correct(0, 0, 1) = -0.000176049;
	gradient_correct(0, 1, 0) = -0.000475683;
	gradient_correct(0, 1, 1) = -0.00416724;
	gradient_correct(1, 0, 0) = -0.0033006;
	gradient_correct(1, 0, 1) = -0.00781465;
	gradient_correct(1, 1, 0) = -0.0126909;
	gradient_correct(1, 1, 1) = 0.02869;

	Tensor<float, 3> gradient_new = *net.gradient_buffer[0];
	Tensor<float, 3> gradient_old = *net.gradient_buffer[1];

	cout << "\nCorrect gradient:\n";
	PrintTensorNumpy(gradient_correct, 2, 2, 2);

	cout << "\nOld gradient:\n";
	PrintTensorNumpy(gradient_old, 1, 1, 8);

	cout << "\nNew gradient:\n";
	PrintTensorNumpy(gradient_new, 2, 2, 2);

	for (Index i = 0; i < 2; i++) {
		for (Index j = 0; j < 2; j++) {
			for (Index k = 0; k < 2; k++) {
				if (abs(gradient_new(i, j, k) - gradient_correct(i, j, k)) > 1e-4) {
					return {name, false};
				}
			}
		}
	}

	return {name, true};
}

pair<string, bool> test_relu_gradient() {
	const string name = "ReLU Gradient";
	cout << "\033[1;33m===== Test " + name + " =====\033[0m\n";

	Tensor<float, 3> *input = new Tensor<float, 3>(2, 2, 2);

	float val = -3.0f;
	for (int i = 0; i < 2; ++i)
		for (int j = 0; j < 2; ++j)
			for (int k = 0; k < 2; ++k)
				(*input)(i, j, k) = val++;

	cout << "Input:" << endl;
	PrintTensorNumpy(*input, 2, 2, 2);

	CNN net(input, 0.1);
	net.add_relu_layer();
	net.add_flatten_layer();
	net.add_softmax_layer();

	net.forward();

	Tensor<float, 3> &output_computed = *net.data_buffer[1];

	cout << "\nOutput:\n";
	PrintTensorNumpy(output_computed, 2, 2, 2);

	cout << "\n\033[1;34mStarting back propagation...\033[0m\n";

	net.backward(3);

	Tensor<float, 3> gradient_correct(2, 2, 2);
	gradient_correct(0, 0, 0) = 0.0;
	gradient_correct(0, 0, 1) = 0.0;
	gradient_correct(0, 1, 0) = 0.0;
	gradient_correct(0, 1, 1) = 0.0;
	gradient_correct(1, 0, 0) = -0.0030291;
	gradient_correct(1, 0, 1) = -0.00713953;
	gradient_correct(1, 1, 0) = -0.0113206;
	gradient_correct(1, 1, 1) = 0.0289801;

	Tensor<float, 3> gradient_new = *net.gradient_buffer[0];
	Tensor<float, 3> gradient_old = *net.gradient_buffer[1];

	cout << "\nCorrect gradient:\n";
	PrintTensorNumpy(gradient_correct, 2, 2, 2);

	cout << "\nOld gradient:\n";
	PrintTensorNumpy(gradient_old, 2, 2, 2);

	cout << "\nNew gradient:\n";
	PrintTensorNumpy(gradient_new, 2, 2, 2);

	for (Index i = 0; i < 2; i++) {
		for (Index j = 0; j < 2; j++) {
			for (Index k = 0; k < 2; k++) {
				if (abs(gradient_new(i, j, k) - gradient_correct(i, j, k)) > 1e-4) {
					return {name, false};
				}
			}
		}
	}

	return {name, true};
}

pair<string, bool> test_maxpool_gradient() {
	const string name = "MaxPooling Gradient";
	cout << "\033[1;33m===== Test " + name + " =====\033[0m\n";

	Tensor<float, 3> *input = new Tensor<float, 3>(1, 4, 4);
	array<float, 16> values = {1, 3, 2, 1, 4, 4, 1, 0, 3, 5, 2, 1, 1, 2, 0, 6};

	for (Index i = 0; i < 16; i++) {
		(*input)(0, i / 4, i % 4) = values[i];
	}

	cout << "Input:" << endl;
	PrintTensorNumpy(*input, 1, 4, 4);

	CNN net(input, 0.1);
	net.add_maxpooling_layer({2, 2}, 2);

	net.forward();

	Tensor<float, 3> &output_computed = *net.data_buffer[1];

	cout << "\nOutput:\n";
	PrintTensorNumpy(output_computed, 1, 2, 2);

	cout << "\n\033[1;34mStarting back propagation...\033[0m\n";

	(*net.gradient_buffer[1])(0, 0, 0) = 10;
	(*net.gradient_buffer[1])(0, 0, 1) = 20;
	(*net.gradient_buffer[1])(0, 1, 0) = 30;
	(*net.gradient_buffer[1])(0, 1, 1) = 40;

	net.backward(3);

	array<float, 16> values_grad = {0, 0, 20, 0, 10, 0, 0, 0, 0, 30, 0, 0, 0, 0, 0, 40};
	Tensor<float, 3> gradient_correct(1, 4, 4);

	for (Index i = 0; i < 16; i++) {
		gradient_correct(0, i / 4, i % 4) = values_grad[i];
	}

	Tensor<float, 3> gradient_new = *net.gradient_buffer[0];
	Tensor<float, 3> gradient_old = *net.gradient_buffer[1];

	cout << "\nCorrect gradient:\n";
	PrintTensorNumpy(gradient_correct, 1, 4, 4);

	cout << "\nOld gradient:\n";
	PrintTensorNumpy(gradient_old, 1, 2, 2);

	cout << "\nNew gradient:\n";
	PrintTensorNumpy(gradient_new, 1, 4, 4);

	for (Index i = 0; i < 1; i++) {
		for (Index j = 0; j < 4; j++) {
			for (Index k = 0; k < 4; k++) {
				if (abs(gradient_new(i, j, k) - gradient_correct(i, j, k)) > 1e-4) {
					return {name, false};
				}
			}
		}
	}

	return {name, true};
}

pair<string, bool> test_fully_connected_training() {
	const string name = "Fully Connected Backward Value";
	cout << "\033[1;33m===== Test " + name + " =====\033[0m\n";

	Tensor<float, 3> *input = new Tensor<float, 3>(1, 1, 4);
	for (int i = 0; i < 4; ++i) {
		(*input)(0, 0, i) = i + 1;
	}

	cout << "Input:\n";
	PrintTensorNumpy(*input, 1, 1, 4);

	float learning_rate = 10;
	CNN net(input, learning_rate);
	net.add_fully_connected_layer(3);
	net.add_softmax_layer();

	auto full_layer_ptr = dynamic_cast<FullyConnectedLayer *>(net.layers[0]);
	auto softmax_layer_ptr = dynamic_cast<SoftMaxLayer *>(net.layers[1]);

	full_layer_ptr->weights << 5, 12, 15, 9, 5, 6, 7, 8, 9, 10, 11, 12;
	full_layer_ptr->biases << 0.3, 0.2, 0.2;

	cout << "\033[1;34m===== Before Training =====\033[0m\n";

	net.forward();

	cout << "\nPredictions:\n";
	for (int i = 0; i < 3; i++) {
		cout << net.predictions[i] << endl;
	}

	cout << "\nWeights:\n";
	PrintMatrixNumpy(full_layer_ptr->weights, 3, 4);
	cout << "\nBiases:\n";
	PrintMatrixNumpy(full_layer_ptr->biases, 3, 1);
	cout << "\nOutput:\n";
	PrintTensorNumpy(*full_layer_ptr->output_data, 1, 1, 3);

	cout << "\033[1;34m===== During Training =====\033[0m\n";

	net.backward(2);

	Tensor input_gradient = *full_layer_ptr->input_gradients;
	Tensor output_gradient = *full_layer_ptr->output_gradients;

	cout << "\nOutput gradient (softmax):\n";
	PrintTensorNumpy(*softmax_layer_ptr->output_gradients, 1, 1, 3);
	cout << "\nInput gradient (softmax):\n";
	PrintTensorNumpy(*softmax_layer_ptr->input_gradients, 1, 1, 3);

	cout << "\nOutput gradient (fully-connected):\n";
	PrintTensorNumpy(output_gradient, 1, 1, 3);
	cout << "\nInput gradient (fully-connected):\n";

	PrintTensorNumpy(input_gradient, 1, 1, 4);
	Tensor<float, 3> input_gradient_correct(1, 1, 4);
	input_gradient_correct(0, 0, 0) = -0.698228;
	input_gradient_correct(0, 0, 1) = 0.349114;
	input_gradient_correct(0, 0, 2) = 0.698227;
	input_gradient_correct(0, 0, 3) = -0.523671;

	MatrixXf weight_gradient_correct(3, 4);
	VectorXf bias_gradient_correct(3);

	weight_gradient_correct << 0.174557, 0.349114, 0.523671, 0.698228, 0, 0, 0, 0, -0.174557, -0.349114, -0.523671, -0.698228;
	bias_gradient_correct << 0.174557, 0, -0.174557;

	cout << "\nWeights gradient:\n";
	PrintMatrixNumpy(full_layer_ptr->weight_gradients, 3, 4);
	cout << "\nBiases gradient:\n";
	PrintMatrixNumpy(full_layer_ptr->bias_gradients, 3, 1);

	cout << endl;
	for (Index i = 0; i < 4; i++) {
		if (abs(input_gradient_correct(0, 0, i) - input_gradient(0, 0, i)) > 1e-4) {
			return {name, false};
		}
	}
	for (Index i = 0; i < 3; i++) {
		for (Index j = 0; j < 4; j++) {
			if (abs(weight_gradient_correct(i, j) - full_layer_ptr->weight_gradients(i, j)) > 1e-4) {
				return {name, false};
			}
		}
	}
	for (Index i = 0; i < 3; i++) {
		if (abs(bias_gradient_correct(i) - full_layer_ptr->bias_gradients(i)) > 1e-4) {
			return {name, false};
		}
	}

	cout << "\033[1;34m===== After Training =====\033[0m\n";

	net.forward();

	cout << "\nWeights:\n";
	PrintMatrixNumpy(full_layer_ptr->weights, 3, 4);
	cout << "\nBiases:\n";
	PrintMatrixNumpy(full_layer_ptr->biases, 3, 1);
	cout << "\nOutput:\n";
	PrintTensorNumpy(*full_layer_ptr->output_data, 1, 1, 3);

	return {name, true};
}

pair<string, bool> test_conv_gradient() {
	const string name = "Convolutional Backward Value";
	cout << "\033[1;33m===== Test " + name + " =====\033[0m\n";

	Tensor<float, 3> *input = new Tensor<float, 3>(2, 2, 2);
	for (Index c = 0; c < 2; ++c) {
		for (Index i = 0; i < 2; ++i) {
			for (Index j = 0; j < 2; ++j) {
				(*input)(c, i, j) = c * 4 + i * 2 + j;
			}
		}
	}

	cout << "Input:\n";
	PrintTensorNumpy(*input, 2, 2, 2);

	float learning_rate = 0.1;
	CNN net(input, learning_rate);
	net.add_convolutional_layer({2, 2}, 2, 1);

	ConvolutionalLayer *conv_layer_ptr = dynamic_cast<ConvolutionalLayer *>(net.layers[0]);

	conv_layer_ptr->kernel << 0.0f, 0.1f, 0.0f, 0.1f, 0.0f, 0.3f, 0.2f, 0.3f, 0.1f, 0.1f, 0.4f, 0.2f, 0.1f, 0.3f, 0.4f, 0.1f;
	conv_layer_ptr->biases << 0.3, 0.2;

	cout << "\033[1;34m===== Before Training =====\033[0m\n";

	net.forward();
	(*net.gradient_buffer[1])(0, 0, 0) = 0.2f;
	(*net.gradient_buffer[1])(1, 0, 0) = -0.2f;

	// cout << "\nPredictions:\n";
	// for (int i = 0; i < 8; i++) {
	// 	cout << net.predictions[i] << endl;
	// }

	cout << "\nKernel:\n";
	PrintMatrixNumpy(conv_layer_ptr->kernel, 2, 8);
	cout << "\nBiases:\n";
	PrintMatrixNumpy(conv_layer_ptr->biases, 2, 2);
	cout << "\nOutput:\n";
	PrintTensorNumpy(*conv_layer_ptr->output_data, 2, 1, 1);

	cout << "\033[1;34m===== During Training =====\033[0m\n";
	net.backward(3);

	Tensor input_gradient = *conv_layer_ptr->input_gradients;
	Tensor output_gradient = *conv_layer_ptr->output_gradients;
	Matrix kernel_gradient = conv_layer_ptr->kernel_gradients;
	Matrix bias_gradient = conv_layer_ptr->bias_gradients;

	// cout << "\nOutput gradient (softmax):\n";
	// PrintTensorNumpy(*softmax_layer_ptr->output_gradient, 1, 1, 8);
	// cout << "\nInput gradient (softmax):\n";
	// PrintTensorNumpy(*softmax_layer_ptr->input_gradient, 1, 1, 8);

	cout << "\nOutput gradient:\n";
	PrintTensorNumpy(output_gradient, 2, 1, 1);
	cout << "\nInput gradient:\n";
	PrintTensorNumpy(input_gradient, 2, 2, 2);
	cout << "\nKernel gradient:\n";
	PrintMatrixNumpy(kernel_gradient, 2, 8);
	cout << "\nBias gradient:\n";
	PrintMatrixNumpy(bias_gradient, 2, 2);

	Tensor<float, 3> input_gradient_correct(2, 2, 2);
	array<float, 8> temp_values = {-0.02f, 0.0f, -0.08, -0.02, -0.02, 0.0, -0.04, 0.04};
	for (Index i = 0; i < 2; ++i) {
		for (Index j = 0; j < 2; ++j) {
			for (Index k = 0; k < 2; ++k) {
				input_gradient_correct(i, j, k) = temp_values[i * 4 + j * 2 + k];
			}
		}
	}

	// Tensor<float, 3> kernel_gradient_correct(2, 2, 2);
	// temp_values = {-0.02f, 0.0f, -0.08, -0.02, -0.02, 0.0, -0.04, 0.04};
	// for (Index i = 0; i < 2; ++i) {
	// 	for (Index j = 0; j < 2; ++j) {
	// 		for (Index k = 0; k < 2; ++k) {
	// 			kernel_gradient_correct(i, j, k) = temp_values[i * 4 + j * 2 + k];
	// 		}
	// 	}
	// }

	for (Index i = 0; i < 2; i++) {
		for (Index j = 0; j < 2; j++) {
			for (Index k = 0; k < 2; k++) {
				if (abs(input_gradient_correct(i, j, k) - input_gradient(i, j, k)) > 1e-4) {
					return {name, false};
				}
			}
		}
	}
	// for (Index i = 0; i < 2; i++) {
	// 	for (Index j = 0; j < 2; j++) {
	// 		for (Index k = 0; k < 2; k++) {
	// 			if (abs(kernel_gradient_correct(i, j, k) - kernel_gradient(i, j * 2 + k)) > 1e-4) {
	// 				return {name, false};
	// 			}
	// 		}
	// 	}
	// }

	cout << "\033[1;34m===== After Training =====\033[0m\n";

	net.forward();

	cout << "\nKernel:\n";
	PrintMatrixNumpy(conv_layer_ptr->kernel, 2, 8);
	cout << "\nBiases:\n";
	PrintMatrixNumpy(conv_layer_ptr->biases, 2, 2);
	cout << "\nOutput:\n";
	PrintTensorNumpy(*conv_layer_ptr->output_data, 2, 1, 1);

	return {name, true};
}

pair<string, bool> test_conv_input_grad_simple() {
	const string name = "Convolutional Backward Value";
	cout << "\033[1;33m===== Test " + name + " =====\033[0m\n";

	Tensor<float, 3> *input = new Tensor<float, 3>(1, 2, 2);
	for (Index i = 0; i < 2; ++i) {
		for (Index j = 0; j < 2; ++j) {
			(*input)(0, i, j) = i * 2 + j;
		}
	}

	cout << "Input:\n";
	PrintTensorNumpy(*input, 1, 2, 2);

	float learning_rate = 10;
	CNN net(input, learning_rate);
	net.add_convolutional_layer({2, 2}, 1, 1);

	ConvolutionalLayer *conv_layer_ptr = dynamic_cast<ConvolutionalLayer *>(net.layers[0]);

	conv_layer_ptr->kernel << 1, 3, 2, 1;
	conv_layer_ptr->biases << 0.3;

	cout << "\033[1;34m===== Before Training =====\033[0m\n";

	net.forward();
	(*net.gradient_buffer.back())(0, 0, 0) = 2.0f;

	// cout << "\nPredictions:\n";
	// for (int i = 0; i < 8; i++) {
	// 	cout << net.predictions[i] << endl;
	// }

	cout << "\nKernel:\n";
	PrintMatrixNumpy(conv_layer_ptr->kernel, 1, 4);
	cout << "\nBiases:\n";
	PrintMatrixNumpy(conv_layer_ptr->biases, 1, 1);
	cout << "\nOutput:\n";
	PrintTensorNumpy(*conv_layer_ptr->output_data, 1, 1, 1);

	cout << "\033[1;34m===== During Training =====\033[0m\n";

	net.backward(3);

	Tensor input_gradient = *conv_layer_ptr->input_gradients;
	Tensor output_gradient = *conv_layer_ptr->output_gradients;

	// cout << "\nOutput gradient (softmax):\n";
	// PrintTensorNumpy(*softmax_layer_ptr->output_gradient, 1, 1, 8);
	// cout << "\nInput gradient (softmax):\n";
	// PrintTensorNumpy(*softmax_layer_ptr->input_gradient, 1, 1, 8);

	cout << "\nOutput gradient:\n";
	PrintTensorNumpy(output_gradient, 1, 1, 1);
	cout << "\nInput gradient:\n";
	PrintTensorNumpy(input_gradient, 1, 2, 2);

	Tensor<float, 3> input_gradient_correct(1, 2, 2);
	array<float, 9> temp_values = {2, 6, 4, 2};
	for (Index i = 0; i < 2; ++i) {
		for (Index j = 0; j < 2; ++j) {
			input_gradient_correct(0, i, j) = temp_values[i * 2 + j];
		}
	}

	// cout << "\nCorrect input gradient:\n";
	// PrintTensorNumpy(input_gradient_correct, 2, 3, 3);

	for (Index i = 0; i < 2; i++) {
		for (Index j = 0; j < 2; j++) {
			if (abs(input_gradient_correct(0, i, j) - input_gradient(0, i, j)) > 1e-4) {
				return {name, false};
			}
		}
	}

	return {name, true};
}

pair<string, bool> test_reshape_back() {
	const string name = "Reshape Backward Value";
	cout << "\033[1;33m===== Test " + name + " =====\033[0m\n";

	Tensor<float, 3> *input = new Tensor<float, 3>(2, 2, 2);

	float val = 1;
	for (int i = 0; i < 2; ++i)
		for (int j = 0; j < 2; ++j)
			for (int k = 0; k < 2; ++k)
				(*input)(i, j, k) = val++;

	cout << "Input:" << endl;
	cout << (*input) << endl;

	CNN net(input, 0.1);
	net.add_flatten_layer();

	net.forward();

	Tensor<float, 3> &output_computed = *net.data_buffer.back();

	cout << "\nOutput:\n";
	cout << output_computed << endl;

	for (int i = 0; i < 8; i++) {
		(*net.gradient_buffer[1])(0, 0, i) = i + 1;
	}

	net.backward(3);

	PrintTensorNumpy(*net.gradient_buffer[0], 2, 2, 2);

	for (int i = 0; i < 2; ++i)
		for (int j = 0; j < 2; ++j)
			for (int k = 0; k < 2; ++k)
				if ((*input)(i, j, k) != (*net.gradient_buffer[0])(i, j, k))
					return {name, false};

	return {name, true};
}

int main() {
	// test_func(test_conv_forward_filters);
	// test_func(test_conv_forward_channels);
	// // test_func(test_conv_forward_stride);
	// // test_func(test_conv_forward_speed);
	//
	// test_func(test_fully_connected_forward_value);
	// test_func(test_fully_connected_forward_speed);
	//
	// test_func(test_maxpool_forward_value);
	// test_func(test_maxpool_forward_speed);
	//
	// test_func(test_relu_forward_value);
	// test_func(test_softmax_forward_value);

	// test_func(test_loss_to_softmax_gradient);
	// test_func(test_softmax_gradient);
	// test_func(test_reshape_gradient);
	// test_func(test_relu_gradient);
	// test_func(test_maxpool_gradient);
	// test_func(test_fully_connected_training);
	// test_func(test_conv_input_grad_simple);
	// test_func(test_conv_gradient);

	// test_func(test_cnn_forward);
	test_func(test_reshape_forward_value);
	test_func(test_reshape_back);

	test_stat();

	return 0;
}
