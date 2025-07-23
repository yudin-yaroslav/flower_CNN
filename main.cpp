#include "layers.hpp"

float get_MSE(Sample *sample, CNN &net) {
	net.set_input_data(sample->image);
	net.forward();

	return net.get_mean_squared_error(sample->label);
}

bool get_correct(Sample *sample, CNN &net) {
	cout << "Testing sample with label " << sample->label << endl;
	net.set_input_data(sample->image);
	net.forward();

	vector<float> predictions = net.predictions;
	auto max_iter = std::max_element(predictions.begin(), predictions.end());
	int index = std::distance(predictions.begin(), max_iter);

	if (index == sample->label) {
		return true;
	}
	return false;
}

float test_CNN(vector<Sample> *testing_data, CNN &net) {
	int sum = 0;
	vector<Sample> &data = *testing_data;

	for (int i = 0; i < data.size(); i++) {
		cout << "Testing image #" << i << endl;
		sum += get_correct(&data[i], net);
	}

	return sum;
}

void train_CNN(vector<Sample> *training_data, CNN &net) {
	vector<Sample> &data = *training_data;

	for (int i = 0; i < data.size(); i++) {
		cout << "Training image #" << i << " with label " << data[i].label << endl;

		net.set_input_data(data[i].image);
		net.forward();
		net.backward(data[i].label);
	}
}

int main() {
	const int C = 3, H = 256, W = 256;
	const int number_of_labels = 4;
	const int number_of_epochs = 10;

	vector<Sample> training_data = load_dataset("../dataset/train", H, W, number_of_labels);
	cout << "Loaded " << training_data.size() << " training samples.\n";

	vector<Sample> testing_data = load_dataset("../dataset/test", H, W, number_of_labels);
	cout << "Loaded " << testing_data.size() << " testing samples.\n";

	CNN net(training_data[0].image, 0.01);

	net.add_convolutional_layer({4, 4}, 16, 4);
	net.add_relu_layer();

	net.add_convolutional_layer({4, 4}, 32, 2);
	net.add_relu_layer();

	net.add_maxpooling_layer({2, 2}, 2);

	net.add_convolutional_layer({3, 3}, 64, 1);
	net.add_relu_layer();

	net.add_maxpooling_layer({2, 2}, 2);

	net.add_convolutional_layer({3, 3}, 128, 1);
	net.add_relu_layer();

	net.add_maxpooling_layer({2, 2}, 2);

	net.add_flatten_layer();
	net.add_fully_connected_layer(8100);
	net.add_fully_connected_layer(number_of_labels);
	net.add_softmax_layer();

	cout << "Initialized CNN." << endl;

	vector<float> accuracy_array(number_of_epochs);

	cout << "\n\n\033[1;34mTest results before training: \033[0m\n";
	test_CNN(&testing_data, net);

	cout << "\n\n\033[1;34mStarting training: \033[0m\n";
	for (int i = 0; i < number_of_epochs; i++) {
		train_CNN(&training_data, net);

		float correct = test_CNN(&testing_data, net);
		float total = static_cast<float>(testing_data.size());
		accuracy_array[i] = correct / total;

		cout << "ACCURACY = " << accuracy_array[i] << endl;
	}

	cout << "\n\033[1;34mTest results after training: \033[0m\n";
	cout << "Accuracy = " << test_CNN(&testing_data, net) / number_of_labels / 5 << endl;

	cout << "\n\033[1;34mPredictions of testing_data: \033[0m\n";
	get_correct(&(testing_data[0]), net);
	for (int i = 0; i < number_of_labels; i++) {
		cout << net.predictions[i] << endl;
	}
	cout << endl;

	get_correct(&(testing_data[1]), net);
	for (int i = 0; i < number_of_labels; i++) {
		cout << net.predictions[i] << endl;
	}
	cout << endl;

	get_correct(&(testing_data[5]), net);
	for (int i = 0; i < number_of_labels; i++) {
		cout << net.predictions[i] << endl;
	}
	cout << endl;

	get_correct(&(testing_data[7]), net);
	for (int i = 0; i < number_of_labels; i++) {
		cout << net.predictions[i] << endl;
	}
	cout << endl;

	cout << "\n\033[1;34mAccuracy array: \033[0m\n";
	for (int i = 0; i < number_of_epochs; i++) {
		cout << accuracy_array[i] << endl;
	}

	// for (int i = 0; i < net.number_of_layers + 1; i++) {
	// 	float *grad_ptr = net.gradient_buffer[i]->data();
	// 	cout << "Gradient #" << i << ": ";
	// 	for (int j = 0; j < 10; j++) {
	// 		cout << grad_ptr[j] << " ";
	// 	}
	// 	cout << endl;
	// }
}
