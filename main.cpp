#include "layers.hpp"

using namespace std;

int main() {
	Matrix<3, int> *input_data = new Matrix<3, int>(3, 3, 3);
	CNN net(input_data);

	net.add_layer<ConvolutionalLayer>(array<size_t, 3>({3, 2, 2}), 96, 1);
	net.print_buffers();
}
