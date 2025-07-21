#include "layers.hpp"

float get_MSE(Sample *sample, CNN *net) {
	net->set_input_data(&(sample->image));
	net->forward();

	return net->get_mean_squared_error(sample->label);
}

bool get_correct(Sample *sample, CNN *net) {
	net->set_input_data(&(sample->image));
	net->forward();

	vector<float> predictions = net->predictions;
	auto max_iter = std::max_element(predictions.begin(), predictions.end());
	int index = std::distance(predictions.begin(), max_iter);

	cout << index << endl;
	if (index == sample->label) {
		return true;
	}
	return false;
}

void test_CNN(vector<Sample> *testing_data, CNN *net) {
	int sum = 0;
	vector<Sample> &data = *testing_data;

	for (int i = 0; i < data.size(); i++) {
		cout << "Testing image #" << i << endl;
		sum += get_correct(&data[i], net);
	}

	cout << sum << " guesses were correct out of " << data.size();

	// float min_error = *min_element(errors.begin(), errors.end());
	// float max_error = *max_element(errors.begin(), errors.end());
	// float average_error = accumulate(errors.begin(), errors.end(), 0.0f) / errors.size();

	// cout << "Min error: " << min_error << endl;
	// cout << "Max error: " << max_error << endl;
	// cout << "Average error: " << average_error << endl;
}

void train_CNN(vector<Sample> *training_data, CNN *net) {
	vector<Sample> &data = *training_data;

	for (int i = 0; i < data.size(); i++) {
		cout << "Training image #" << i << endl;

		net->set_input_data(&(data[i].image));
		net->forward();
		net->backward(data[i].label);
	}
}

int main() {
	const int C = 3, H = 256, W = 256;

	vector<Sample> training_data = load_dataset("../dataset/train", H, W, 105);
	cout << "Loaded " << training_data.size() << " training samples.\n";

	vector<Sample> testing_data = load_dataset("../dataset/test", H, W, 10);
	cout << "Loaded " << testing_data.size() << " testing samples.\n";

	CNN *net = new CNN(&(training_data[0].image), 10.0);

	net->add_convolutional_layer({4, 4}, 10, 1); // -> 254x254x32
	net->add_relu_layer();
	net->add_maxpooling_layer({2, 2}, 2); // -> 127x127x32

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

	net->add_flatten_layer();			 // -> 14*14*256 = 50176
	net->add_fully_connected_layer(128); // ~25M params here!
	net->add_fully_connected_layer(102); // final
	net->add_softmax_layer();

	cout << "Initialized CNN." << endl;

	cout << "\n\n\033[1;34mMSE: Test results before training: \033[0m\n";
	test_CNN(&testing_data, net);

	cout << "\n\n\033[1;34mMSE: Starting training: \033[0m\n";
	train_CNN(&training_data, net);

	// cout << "\n\033[1;34mMSE: Test results after training: \033[0m\n";
	// test_CNN(&testing_data, net);
	//
	// cout << "\n\033[1;34mMSE: Predictions of testing_data: \033[0m\n";
	// get_correct(&(testing_data[0]), net);
	// float test_1 = net->predictions[0];
	// for (int i = 0; i < 4; i++) {
	// 	cout << net->predictions[i] << endl;
	// }
	// cout << endl;
	//
	// get_correct(&(testing_data[1]), net);
	// float test_2 = net->predictions[0];
	// for (int i = 0; i < 4; i++) {
	// 	cout << net->predictions[i] << endl;
	// }
	// cout << endl;
	//
	// get_correct(&(testing_data[5]), net);
	// for (int i = 0; i < 4; i++) {
	// 	cout << net->predictions[i] << endl;
	// }
	// cout << endl;
	//
	// get_correct(&(testing_data[7]), net);
	// for (int i = 0; i < 4; i++) {
	// 	cout << net->predictions[i] << endl;
	// }
	// cout << endl;

	for (int i = 0; i < net->number_of_layers + 1; i++) {
		float *grad_ptr = net->gradient_buffer[i]->data();
		cout << "Gradient #" << i << ": ";
		for (int j = 0; j < 10; j++) {
			cout << grad_ptr[j] << " ";
		}
		cout << endl;
	}
}
