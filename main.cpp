#include "layers.hpp"

using namespace std;

int main() {
	Matrix<3, int> *input_data = new Matrix<3, int>(3, 3, 3);
	CNN net(input_data);

	net.add_layer<ConvolutionalLayer>(array<size_t, 2>({2, 2}), 2, 1);
	// net.print_buffers();

	auto conv_layer_ptr = dynamic_cast<ConvolutionalLayer *>(net.layers[0]);
	cout << "kernel:" << endl;
	conv_layer_ptr->kernel[0].print();
	conv_layer_ptr->kernel[1].print();

	cout << "biases:" << endl;
	conv_layer_ptr->biases[0].print();
	conv_layer_ptr->biases[1].print();

	(*net.layers[0]).forward();
	net.print_buffers();
}
