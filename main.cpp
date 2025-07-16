#include "layers.hpp"
#include <chrono>
using namespace std;

void test_convolutional_layer() {
	cout << "\n=== Test: Convolutional Layer ===\n";

	Matrix<3, float> *input_data = new Matrix<3, float>(1, 3, 3);
	(*input_data)(0, 0, 0) = 1;
	(*input_data)(0, 0, 1) = 0;
	(*input_data)(0, 0, 2) = -1;
	(*input_data)(0, 1, 0) = 2;
	(*input_data)(0, 1, 1) = 3;
	(*input_data)(0, 1, 2) = 4;
	(*input_data)(0, 2, 0) = -1;
	(*input_data)(0, 2, 1) = -2;
	(*input_data)(0, 2, 2) = -3;

	CNN net(input_data);
	net.add_convolutional_layer({2, 2}, 1, 1); // 1 filter, stride 1

	auto conv_layer_ptr = dynamic_cast<ConvolutionalLayer *>(net.layers[0]);
	cout << "Kernel:\n";
	conv_layer_ptr->kernel[0].print();

	cout << "Biases:\n";
	conv_layer_ptr->biases[0].print();

	conv_layer_ptr->forward();

	cout << "Input:\n";
	input_data->print();

	cout << "Output after Convolution:\n";
	net.buffers.back()->print();
}

void test_max_pooling_layer() {
	cout << "\n=== Test: MaxPooling Layer ===\n";

	Matrix<3, float> *input_data = new Matrix<3, float>(1, 3, 3);
	(*input_data)(0, 0, 0) = 1;
	(*input_data)(0, 0, 1) = 3;
	(*input_data)(0, 0, 2) = 2;
	(*input_data)(0, 1, 0) = 4;
	(*input_data)(0, 1, 1) = 6;
	(*input_data)(0, 1, 2) = 5;
	(*input_data)(0, 2, 0) = 7;
	(*input_data)(0, 2, 1) = 9;
	(*input_data)(0, 2, 2) = 8;

	CNN net(input_data);
	net.add_pooling_layer({2, 2}, 1);

	auto pool_layer_ptr = dynamic_cast<MaxPoolingLayer *>(net.layers[0]);
	pool_layer_ptr->forward();

	cout << "Input:\n";
	input_data->print();

	cout << "Output after MaxPooling:\n";
	net.buffers.back()->print();
}

void test_fully_connected_layer() {
	cout << "\n=== Test: Fully Connected Layer ===\n";

	Matrix<3, float> *input_data = new Matrix<3, float>(1, 3, 1);
	(*input_data)(0, 0, 0) = 5;
	(*input_data)(0, 1, 0) = -1;
	(*input_data)(0, 2, 0) = 8;

	CNN net(input_data);
	net.add_fully_connected_layer(2); // output dim = 2

	auto fc_layer_ptr = dynamic_cast<FullyConnectedLayer *>(net.layers[0]);
	cout << "Weights:\n";
	fc_layer_ptr->weights.print();

	fc_layer_ptr->forward();

	cout << "Input:\n";
	input_data->print();

	cout << "Output after FC layer:\n";
	net.buffers.back()->print();
}

void test_relu_layer() {
	cout << "\n=== Test: ReLU Layer ===\n";

	Matrix<3, float> *input_data = new Matrix<3, float>(1, 2, 3);
	(*input_data)(0, 0, 0) = -2;
	(*input_data)(0, 0, 1) = 0;
	(*input_data)(0, 0, 2) = 5;
	(*input_data)(0, 1, 0) = -1;
	(*input_data)(0, 1, 1) = 3;
	(*input_data)(0, 1, 2) = -4;

	CNN net(input_data);
	net.add_relu_layer();

	auto relu_layer_ptr = dynamic_cast<ReLULayer *>(net.layers[0]);
	relu_layer_ptr->forward();

	cout << "Input:\n";
	input_data->print();

	cout << "Output after ReLU:\n";
	net.buffers.back()->print();
}

void test_reshape_layer() {
	cout << "\n=== Test: Reshape Layer ===\n";

	Matrix<3, float> *input_data = new Matrix<3, float>(1, 2, 3);
	int val = 0;
	for (size_t i = 0; i < 1; i++)
		for (size_t j = 0; j < 2; j++)
			for (size_t k = 0; k < 3; k++)
				(*input_data)(i, j, k) = val++;

	CNN net(input_data);
	net.add_reshape_layer({1, 6, 1});

	auto reshape_layer_ptr = dynamic_cast<ReshapeLayer *>(net.layers[0]);
	reshape_layer_ptr->forward();

	cout << "Input:\n";
	input_data->print();

	cout << "Output after reshape (1, 6, 1):\n";
	net.buffers.back()->print();
}

void test_cnn_forward() {
	cout << "\n=== Test: CNN Forward Pass ===\n";

	Matrix<3, float> *input_data = new Matrix<3, float>(3, 256, 256);

	for (size_t c = 0; c < 3; c++) {
		for (size_t i = 0; i < 256; i++) {
			for (size_t j = 0; j < 256; j++) {
				(*input_data)(c, i, j) = static_cast<float>((c + i + j) % 10);
			}
		}
	}

	CNN net(input_data);

	net.add_convolutional_layer({12, 12}, 96, 4);
	net.add_relu_layer();

	net.add_pooling_layer({3, 3}, 2);

	net.add_convolutional_layer({5, 5}, 256, 1);
	net.add_relu_layer();

	net.add_convolutional_layer({3, 3}, 256, 1);
	net.add_relu_layer();
	net.add_convolutional_layer({3, 3}, 256, 1);
	net.add_relu_layer();
	net.add_convolutional_layer({3, 3}, 96, 1);
	net.add_relu_layer();

	net.add_pooling_layer({3, 3}, 2);

	net.add_reshape_layer();
	net.add_fully_connected_layer(512);
	net.add_fully_connected_layer(512);
	net.add_fully_connected_layer(102);
	net.add_relu_layer();

	cout << "Starting forward propagation..." << endl;
	net.forward();

	// Print final output shape and some values
	auto final_output = net.buffers.back();
	cout << "Final output shape: (" << final_output->get_sizes()[0] << ", " << final_output->get_sizes()[1] << ", "
		 << final_output->get_sizes()[2] << ")\n";

	cout << "Final output (first 10 elements): ";
	for (size_t i = 0; i < 10 && i < final_output->data().size(); i++) {
		cout << final_output->data()[i] << " ";
	}
	cout << "\n";
}

int main() {
	// test_convolutional_layer();
	// test_max_pooling_layer();
	// test_fully_connected_layer();
	// test_relu_layer();
	// test_reshape_layer();

	test_cnn_forward();

	return 0;
}
