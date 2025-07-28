#include "layers.hpp"

vector<bool> get_correct(Batch *sample, CNN &net) {
	int batch_size = sample->labels.size();

	cout << "Testing sample batch: batch_size = " << batch_size << ", labels: ";
	for (Index b = 0; b < batch_size; b++) {
		cout << sample->labels[b] << " ";
	}
	cout << endl;

	net.set_input_data(sample->images);
	net.forward();

	vector<bool> result(batch_size);
	for (Index b = 0; b < batch_size; b++) {
		MatrixXf predictions = net.predictions;

		float max_value = 0.0f;
		Index max_index = 0;

		for (Index p = 0; p < net.number_of_predictions; p++) {
			if (max_value < predictions(b, p)) {
				max_value = predictions(b, p);
				max_index = p;
			}
		}

		if (max_index == sample->labels[b]) {
			result[b] = true;
		} else {
			result[b] = false;
		}
	}

	return result;
}

float test_CNN(vector<Batch> *testing_data, CNN &net) {
	const int batch_size = (*testing_data)[0].labels.size();

	int sum = 0;
	for (int i = 0; i < testing_data->size(); i++) {
		cout << "Testing batch #" << i << endl;

		vector<bool> result = get_correct(&(*testing_data)[i], net);
		for (int b = 0; b < batch_size; b++) {
			sum += result[b];
		}
	}

	return sum;
}

void train_CNN(vector<Batch> *training_data, CNN &net) {
	for (int i = 0; i < training_data->size(); i++) {
		cout << "Training batch #" << i << endl;

		net.set_input_data((*training_data)[i].images);
		net.forward();
		net.backward((*training_data)[i].labels);
	}
}

int main() {
	const int C = 3, H = 64, W = 64;
	const int number_of_labels = 5;
	const int number_of_epochs = 5;

	const int batch_size = 5;

	vector<Batch> training_data = load_dataset("../dataset/train", H, W, number_of_labels, batch_size);
	cout << "Loaded " << training_data.size() << " training batches.\n";

	vector<Batch> testing_data = load_dataset("../dataset/test", H, W, number_of_labels, batch_size);
	cout << "Loaded " << testing_data.size() << " testing batches.\n";

	CNN net(training_data[0].images, 1e-2 * 2, batch_size);

	net.add_convolutional_layer({4, 4}, 32, 4); // 64x64 → 16x16
	net.add_leaky_relu_layer();

	net.add_convolutional_layer({2, 2}, 64, 2); // 16x16 → 8x8
	net.add_leaky_relu_layer();

	net.add_convolutional_layer({2, 2}, 128, 2); // 8x8 → 4x4
	net.add_leaky_relu_layer();

	net.add_flatten_layer();
	net.add_fully_connected_layer(256);
	net.add_dropout_layer(0.5f);
	net.add_fully_connected_layer(number_of_labels);
	net.add_softmax_layer();

	cout << "Initialized CNN." << endl;

	vector<float> accuracy_array(number_of_epochs);

	cout << "\n\n\033[1;34mTest results before training: \033[0m\n";
	test_CNN(&testing_data, net);

	cout << "\n\033[1;34mStarting training: \033[0m\n";
	for (int i = 0; i < number_of_epochs; i++) {
		train_CNN(&training_data, net);

		float correct = test_CNN(&testing_data, net);
		float total = static_cast<float>(testing_data.size() * batch_size);
		accuracy_array[i] = correct / total;

		cout << "\033[1;34mAccuracy: \033[0m = " << accuracy_array[i] << endl;

		// cout << "\n\033[1;34mPredictions of testing_data: \033[0m\n";
		// get_correct(&(testing_data[0]), net);
		// for (int i = 0; i < number_of_labels; i++) {
		// 	cout << net.predictions.data()[i] << endl;
		// }
		// cout << endl;
		//
		// get_correct(&(testing_data[1]), net);
		// for (int i = 0; i < number_of_labels; i++) {
		// 	cout << net.predictions.data()[i] << endl;
		// }
		// cout << endl;
		//
		// get_correct(&(testing_data[5]), net);
		// for (int i = 0; i < number_of_labels; i++) {
		// 	cout << net.predictions[i] << endl;
		// }
		// cout << endl;
		//
		// get_correct(&(testing_data[7]), net);
		// for (int i = 0; i < number_of_labels; i++) {
		// 	cout << net.predictions[i] << endl;
		// }
		// cout << endl;
	}

	cout << "\n\033[1;34mTest results after training: \033[0m\n";
	cout << "Accuracy = " << test_CNN(&testing_data, net) / testing_data.size() << endl;

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
