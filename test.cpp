#include "layers.hpp"
#include <bits/stdc++.h>
#include <chrono>

using namespace std::chrono;
using std::cout, std::endl;

void test_func(std::function<bool()> func) {
	auto t1 = high_resolution_clock::now();
	bool result = func();
	auto t2 = high_resolution_clock::now();

	auto ms_int = duration_cast<milliseconds>(t2 - t1);
	cout << endl << "Time duration:" << endl;
	cout << ms_int.count() << "ms\n";

	if (result == true && ms_int.count() < 1000) {
		cout << "\n\033[1;32mTest passed ✔️\033[0m\n\n";
	} else {
		cout << "\n\033[1;31mTest failed ❌\033[0m\n\n";
	}
}

bool test_conv_forward_value() {
	cout << "\033[1;33m===== Test Convolutional Forward Value =====\033[0m\n";

	Tensor<float, 3> *input = new Tensor<float, 3>(2, 3, 2);

	float val = 1;
	for (Index i = 0; i < 2; ++i)
		for (Index j = 0; j < 3; ++j)
			for (Index k = 0; k < 2; ++k)
				(*input)(i, j, k) = val++;

	cout << "Input:" << endl;
	cout << (*input) << endl;

	CNN net(input);

	net.add_convolutional_layer(array<Index, 2>{2, 2}, 2, 1);

	auto layer_ptr = dynamic_cast<ConvolutionalLayer *>(net.layers[0]);

	cout << endl << "Kernel:" << endl;
	for (Index f = 0; f < layer_ptr->filters; ++f) {
		for (Index c = 0; c < layer_ptr->channels; ++c) {
			for (Index x = 0; x < layer_ptr->kernel_sizes[0]; ++x)
				for (Index y = 0; y < layer_ptr->kernel_sizes[1]; ++y)
					layer_ptr->kernel(f, c, x, y) = 100 * f + 10 * c + x + 0.1 * y;
		}
		cout << layer_ptr->kernel(f) << endl;
	}

	cout << endl << "Biases:" << endl;
	for (Index i = 0; i < layer_ptr->filters; ++i) {
		for (Index j = 0; j < layer_ptr->output_sizes[1]; ++j)
			for (Index k = 0; k < layer_ptr->output_sizes[2]; ++k)
				layer_ptr->biases(i, j, k) = i + j + k;
		cout << layer_ptr->biases(i) << endl;
	}

	net.forward();

	Tensor<float, 3> output_correct(2, 2, 1);
	output_correct(0, 0, 0) = 368.4;
	output_correct(0, 1, 0) = 4769.4;
	output_correct(1, 0, 0) = 458.2;
	output_correct(1, 1, 0) = 6459.2;

	Tensor<float, 3> output_computed = *net.buffers.back();

	cout << endl << "Output:" << endl;
	cout << (*net.buffers.back()) << endl;

	for (Index i = 0; i < 2; ++i) {
		for (Index j = 0; j < 2; ++j) {
			for (Index k = 0; k < 1; ++k) {
				if (output_correct(i, j, k) != output_computed(i, j, k)) {
					return false;
				}
			}
		}
	}

	return true;
}

bool test_conv_forward_speed() {
	cout << "\033[1;33m===== Test Convolutional Forward Speed =====\033[0m\n";

	Tensor<float, 3> *input = new Tensor<float, 3>(3, 256, 256);

	int val = 1;
	for (Index i = 0; i < 3; ++i)
		for (Index j = 0; j < 256; ++j)
			for (Index k = 0; k < 256; ++k)
				(*input)(i, j, k) = static_cast<float>((val++) % 16) / 16;

	CNN net(input);

	net.add_convolutional_layer(array<Index, 2>{8, 8}, 96, 4);
	net.forward();

	return true;
}

bool test_fully_connected_forward_value() {
	cout << "\033[1;33m===== Test Fully Connected Forward Value =====\033[0m\n";

	Tensor<float, 3> *input = new Tensor<float, 3>(1, 1, 4);
	for (int i = 0; i < 4; ++i) {
		(*input)(0, 0, i) = i + 1;
	}

	cout << "Input:\n";
	cout << *input << endl;

	CNN net(input);
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

	Tensor<float, 3> &output_computed = *net.buffers.back();

	cout << "\nOutput:\n";
	cout << output_computed << endl;

	for (int i = 0; i < 3; ++i) {
		if (output_correct(0, 0, i) != output_computed(0, 0, i)) {
			return false;
		}
	}

	return true;
}

bool test_fully_connected_forward_speed() {
	cout << "\033[1;33m===== Test Fully Connected Forward Speed =====\033[0m\n";

	Tensor<float, 3> *input = new Tensor<float, 3>(1, 1, 1024);
	for (int i = 0; i < 4; ++i) {
		(*input)(0, 0, i) = i + 1;
	}

	CNN net(input);
	net.add_fully_connected_layer(1024);

	net.forward();

	return true;
}

bool test_maxpool_forward_value() {
	cout << "\033[1;33m===== Test Max Pooling Forward Value =====\033[0m\n";

	Tensor<float, 3> *input = new Tensor<float, 3>(1, 4, 4);

	float val = 1;
	for (int i = 0; i < 4; ++i)
		for (int j = 0; j < 4; ++j)
			(*input)(0, i, j) = val++;

	cout << "\nInput:\n";
	cout << (*input) << endl;

	CNN net(input);

	net.add_maxpooling_layer(array<Index, 2>{2, 2}, 2);

	net.forward();

	Tensor<float, 3> output_computed = *net.buffers.back();

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
				return false;

	return true;
}

bool test_maxpool_forward_speed() {
	cout << "\033[1;33m===== Test Max Pooling Forward Speed =====\033[0m\n";

	Tensor<float, 3> *input = new Tensor<float, 3>(3, 256, 256);

	for (int c = 0; c < 3; ++c)
		for (int i = 0; i < 256; ++i)
			for (int j = 0; j < 256; ++j)
				(*input)(c, i, j) = rand() % 100;

	CNN net(input);

	net.add_maxpooling_layer(array<Index, 2>{2, 2}, 2);

	net.forward();

	return true;
}

bool test_relu_forward_value() {
	cout << "\033[1;33m===== Test ReLU Forward Value =====\033[0m\n";

	Tensor<float, 3> *input = new Tensor<float, 3>(2, 2, 2);

	float val = 1;
	for (int i = 0; i < 2; ++i)
		for (int j = 0; j < 2; ++j)
			for (int k = 0; k < 2; ++k)
				(*input)(i, j, k) = -4 + val++;

	cout << "\nInput:\n";
	cout << (*input) << endl;

	CNN net(input);

	net.add_relu_layer();

	net.forward();

	Tensor<float, 3> output_computed = *net.buffers.back();

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
					return false;

	return true;
}

bool test_softmax_forward_value() {
	cout << "\033[1;33m===== Test SoftMax Forward Value =====\033[0m\n";

	Tensor<float, 3> *input = new Tensor<float, 3>(1, 1, 4);
	for (int i = 0; i < 4; ++i) {
		(*input)(0, 0, i) = i + 1;
	}

	cout << "Input:\n";
	cout << *input << endl;

	CNN net(input);
	net.add_softmax_layer();

	net.forward();

	Tensor<float, 3> output_correct(1, 1, 4);
	output_correct(0, 0, 0) = 0.0320;
	output_correct(0, 0, 1) = 0.0871;
	output_correct(0, 0, 2) = 0.2369;
	output_correct(0, 0, 3) = 0.6439;

	Tensor<float, 3> &output_computed = *net.buffers.back();

	cout << "\nOutput:\n";
	cout << output_computed << endl;

	for (int i = 0; i < 4; ++i) {
		if (fabs(output_correct(0, 0, i) - output_computed(0, 0, i)) > 1e-4) {
			return false;
		}
	}

	return true;
}

bool test_reshape_forward_value() {
	Tensor<float, 3> *input = new Tensor<float, 3>(2, 2, 2);

	float val = 1;
	for (int i = 0; i < 2; ++i)
		for (int j = 0; j < 2; ++j)
			for (int k = 0; k < 2; ++k)
				(*input)(i, j, k) = val++;

	cout << "Input:" << endl;
	cout << (*input) << endl;

	CNN net(input);
	net.add_flatten_layer();

	net.forward();

	Tensor<float, 3> output_correct(1, 1, 8);
	for (int i = 0; i < 8; i++) {
		output_correct(0, 0, i) = i + 1;
	}

	Tensor<float, 3> &output_computed = *net.buffers.back();

	cout << "\nOutput:\n";
	cout << output_computed << endl;
	cout << output_correct << endl;

	for (int i = 0; i < 8; ++i) {
		if (output_correct(0, 0, i) != output_computed(0, 0, i)) {
			return false;
		}
	}

	return true;
}

bool test_cnn_forward() {
	cout << "\033[1;33m===== Test CNN Forward =====\033[0m\n";

	Tensor<float, 3> *input = new Tensor<float, 3>(3, 256, 256);

	int val = 1;
	for (Index i = 0; i < 3; ++i)
		for (Index j = 0; j < 256; ++j)
			for (Index k = 0; k < 256; ++k)
				(*input)(i, j, k) = static_cast<float>((val++) % 16) / 16;

	CNN net(input);

	net.add_convolutional_layer({12, 12}, 96, 4);
	net.add_relu_layer();
	net.add_softmax_layer();

	net.add_maxpooling_layer({3, 3}, 2);

	net.add_convolutional_layer({5, 5}, 256, 1);
	net.add_relu_layer();
	net.add_softmax_layer();
	net.add_convolutional_layer({3, 3}, 256, 1);
	net.add_relu_layer();
	net.add_softmax_layer();
	net.add_convolutional_layer({3, 3}, 96, 1);
	net.add_relu_layer();
	net.add_softmax_layer();

	net.add_maxpooling_layer({3, 3}, 2);

	net.add_flatten_layer();
	net.add_fully_connected_layer(512);
	net.add_fully_connected_layer(216);
	net.add_fully_connected_layer(102);
	net.add_softmax_layer();

	cout << "First 10 input values: ";
	for (Index j = 0; j < 10; ++j) {
		cout << (*net.buffers[0])(0, 0, j) << " ";
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

		auto output = net.buffers[i + 1];
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
	auto result = net.buffers.back();
	for (Index j = 0; j < 102; ++j) {
		float value = (*result)(0, 0, j);
		if (value > 0.01) {
			cout << "result[" << j << "] = " << value << endl;
		}
		sum += value;
	}

	cout << "\nSum: " << sum << endl;

	return true;
}

int main() {
	// test_func(test_conv_forward_value);
	// test_func(test_conv_forward_speed);
	//
	// test_func(test_fully_connected_forward_value);
	// test_func(test_fully_connected_forward_speed);
	//
	// test_func(test_maxpool_forward_value);
	// test_func(test_maxpool_forward_speed);
	//
	// test_func(test_relu_forward_value);
	// test_func(test_softmax_forward_value);
	// test_func(test_reshape_forward_value);

	test_func(test_cnn_forward);

	return 0;
}
