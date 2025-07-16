#include "layers.hpp"
#include <chrono>

using namespace std::chrono;
using std::cout, std::endl;

void test_conv_layer() {
	cout << "===== Test Convolutional Layer =====" << endl;
	auto t1 = high_resolution_clock::now();

	Tensor<float, 3> *input = new Tensor<float, 3>(3, 3, 3);

	float val = 1;
	for (int i = 0; i < 3; ++i)
		for (int j = 0; j < 3; ++j)
			for (int k = 0; k < 3; ++k)
				(*input)(i, j, k) = val++;

	cout << "Input:" << endl;
	cout << (*input) << endl;

	CNN net(input);

	net.add_convolutional_layer(array<Index, 2>{2, 2}, 2, 1);

	auto layer_ptr = dynamic_cast<ConvolutionalLayer *>(net.layers[0]);

	cout << endl << "Kernel:" << endl;
	for (int i = 0; i < layer_ptr->filters; ++i) {
		for (int j = 0; j < layer_ptr->kernel_sizes[1]; ++j)
			for (int k = 0; k < layer_ptr->kernel_sizes[2]; ++k)
				layer_ptr->kernel[i](j, k) = 100 * i + 10 * j + k;
		cout << layer_ptr->kernel[i] << endl;
	}

	cout << endl << "Biases:" << endl;
	for (int i = 0; i < layer_ptr->filters; ++i) {
		for (int j = 0; j < layer_ptr->output_sizes[1]; ++j)
			for (int k = 0; k < layer_ptr->output_sizes[2]; ++k)
				layer_ptr->biases[i](j, k) = i + j + k;
		cout << layer_ptr->biases[i] << endl;
	}

	net.forward();

	Tensor<float, 3> output_correct(2, 2, 2);
	output_correct(0, 0, 0) = 885;
	output_correct(0, 0, 1) = 952;
	output_correct(0, 1, 0) = 1084;
	output_correct(0, 1, 1) = 1151;
	output_correct(1, 0, 0) = 15286;
	output_correct(1, 0, 1) = 16553;
	output_correct(1, 1, 0) = 19085;
	output_correct(1, 1, 1) = 20352;

	Tensor<float, 3> output_computed = *net.buffers.back();

	cout << endl << "Output:" << endl;
	cout << (*net.buffers.back()) << endl;

	auto t2 = high_resolution_clock::now();
	auto ms_int = duration_cast<milliseconds>(t2 - t1);
	cout << endl << "Time duration:" << endl;
	cout << ms_int.count() << "ms\n";

	for (int i = 0; i < 2; ++i) {
		for (int j = 0; j < 2; ++j) {
			for (int k = 0; k < 2; ++k) {
				if (output_correct(i, j, k) != output_computed(i, j, k)) {
					cout << endl << "Test failed ❌" << endl;
					return;
				}
			}
		}
	}

	cout << endl << "Test passed ✔️" << endl;
}

int main() {
	test_conv_layer();

	return 0;
}
