#include <filesystem>
#include <iostream>
#include <map>
#include <opencv2/opencv.hpp>
#include <random>
#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>

namespace fs = std::filesystem;
using std::vector, std::string, std::map;
using namespace cv;
using namespace Eigen;

struct Sample {
	Tensor<float, 3> image;
	int label;
};

struct Batch {
	Tensor<float, 4> images;
	vector<int> labels;
};

inline Tensor<float, 3> load_image(const std::string &path, int H, int W) {
	Mat img = imread(path, IMREAD_COLOR);
	if (img.empty()) {
		std::cerr << "Failed to load " << path << std::endl;
	}
	cvtColor(img, img, COLOR_BGR2RGB);
	resize(img, img, Size(W, H));

	Tensor<float, 3> tensor(3, H, W);
	for (int h = 0; h < H; ++h)
		for (int w = 0; w < W; ++w)
			for (int c = 0; c < 3; ++c)
				tensor(c, h, w) = static_cast<float>(img.at<Vec3b>(h, w)[c]) / 255.0f;

	return tensor;
}

inline vector<Batch> load_dataset(const string &root_dir, int H, int W, int number_of_labels, int batch_size) {
	vector<Sample> flat_dataset;
	map<string, int> class_to_label;

	int label_counter = 0;

	for (const auto &entry : fs::directory_iterator(root_dir)) {

		if (!entry.is_directory())
			continue;

		if (label_counter >= number_of_labels)
			break;

		string class_name = entry.path().filename().string();
		int label = label_counter++;
		class_to_label[class_name] = label;

		for (const auto &img_entry : fs::directory_iterator(entry)) {
			if (!img_entry.is_regular_file())
				continue;

			string img_path = img_entry.path().string();
			Tensor<float, 3> img_tensor = load_image(img_path, H, W);

			flat_dataset.push_back({img_tensor, label});
		}
	}
	std::random_device rd;
	std::mt19937 g(rd());
	std::shuffle(flat_dataset.begin(), flat_dataset.end(), g);

	std::cout << flat_dataset.size() << std::endl;

	int total_samples = (flat_dataset.size() / batch_size) * batch_size;
	int total_batches = total_samples / batch_size;

	vector<Batch> dataset;

	for (int b = 0; b < total_batches; b++) {
		Tensor<float, 4> batched_images(batch_size, 3, H, W); // (B, C, H, W)
		vector<int> batched_labels(batch_size);

		for (Index i = 0; i < batch_size; ++i) {
			const auto &sample = flat_dataset[b * batch_size + i];
			for (Index c = 0; c < 3; ++c)
				for (Index h = 0; h < H; ++h)
					for (Index w = 0; w < W; ++w)
						batched_images(i, c, h, w) = sample.image(c, h, w);
			batched_labels[i] = sample.label;
		}

		dataset.push_back({batched_images, batched_labels});
	}

	return dataset;
}
