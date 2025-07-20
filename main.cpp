#include "layers.hpp"

vector<float> get_probabilities(Sample *sample, CNN *net) {
	net->set_input_data(&(sample->image));
	net->forward();

	Tensor<float, 3> *prob_tensor = net->data_buffer.back();

	vector<float> result(prob_tensor->size());
	for (int i = 0; i < prob_tensor->size(); ++i)
		result[i] = (*prob_tensor)(i);

	return result;
}

int main() {
	string dataset_path = "../dataset/train";
	const int C = 3, H = 256, W = 256;

	vector<Sample> training_data = load_dataset(dataset_path, H, W);
	cout << "Loaded " << training_data.size() << " training samples.\n";

	CNN net({C, H, W}, 0.1);

	net.add_convolutional_layer({12, 12}, 96, 4);
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

	vector<float> result;
	for (int i = 0; i < 1000; i++) {
		cout << "Image #" << i << endl;
		result = get_probabilities(&training_data[i], &net);
	}

	cout << "\n\033[1;34mResults bigger than 0.01: \033[0m\n";

	for (Index j = 0; j < 102; ++j) {
		// if (value > 0.01) {
		cout << "result[" << j << "] = " << result[j] << endl;
		// }
	}
}
