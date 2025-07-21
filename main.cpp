#include "layers.hpp"
#include <algorithm>
#include <numeric>

float forward(Sample *sample, CNN *net) {
	net->set_input_data(&(sample->image));
	net->forward();

	return net->get_mean_squared_error(sample->label);
}

void test_CNN(vector<Sample> *testing_data, CNN *net) {
	vector<float> errors;
	vector<Sample> &data = *testing_data;

	for (int i = 0; i < data.size(); i++) {
		cout << "Testing image #" << i << endl;
		errors.push_back(forward(&data[i], net));
	}

	// float min_error = *min_element(errors.begin(), errors.end());
	// float max_error = *max_element(errors.begin(), errors.end());
	// float average_error = accumulate(errors.begin(), errors.end(), 0.0f) / errors.size();
	//
	// cout << "Min error: " << min_error << endl;
	// cout << "Max error: " << max_error << endl;
	// cout << "Average error: " << average_error << endl;
}

int main() {
	const int C = 3, H = 256, W = 256;

	vector<Sample> training_data = load_dataset("../dataset/train", H, W, 100);
	cout << "Loaded " << training_data.size() << " training samples.\n";

	vector<Sample> testing_data = load_dataset("../dataset/test", H, W, 100);
	cout << "Loaded " << testing_data.size() << " testing samples.\n";

	CNN net(&(training_data[0].image), 100);

	net.add_convolutional_layer({3, 3}, 10, 1); // -> 254x254x32
	net.add_relu_layer();
	net.add_maxpooling_layer({2, 2}, 2); // -> 127x127x32

	// net.add_convolutional_layer({3, 3}, 64, 1); // -> 125x125x64
	// net.add_relu_layer();
	// net.add_maxpooling_layer({2, 2}, 2); // -> 62x62x64

	// net.add_convolutional_layer({3, 3}, 128, 1); // -> 60x60x128
	// net.add_relu_layer();
	// net.add_maxpooling_layer({2, 2}, 2); // -> 30x30x128
	//
	// net.add_convolutional_layer({3, 3}, 256, 1); // -> 28x28x256
	// net.add_relu_layer();
	// net.add_maxpooling_layer({2, 2}, 2); // -> 14x14x256

	net.add_flatten_layer();			// -> 14*14*256 = 50176
	net.add_fully_connected_layer(102); // ~25M params here!
	net.add_fully_connected_layer(102); // final
	net.add_softmax_layer();

	cout << "Initialized CNN." << endl;

	cout << "\n\n\033[1;34mMSE: Test results before training: \033[0m\n";
	test_CNN(&testing_data, &net);

	// for (int i = 0; i < 40; i++) {
	// 	float result_before = forward(&training_data[0], &net);
	// 	cout << "\n\033[1;34mMSE: " << result_before << " \033[0m\n";
	//
	// 	cout << "Cycle #" << i << endl;
	// 	net.backward(training_data[0].label);
	// }
	//
	// cout << "\n\033[1;34mMSE: Test results after training: \033[0m\n";
	// test_CNN(&testing_data, &net);
}
