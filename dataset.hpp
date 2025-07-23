#include <filesystem>
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

Tensor<float, 3> load_image(const std::string &path, int H, int W) {
	Mat img = imread(path, IMREAD_COLOR);
	if (img.empty()) {
		throw std::runtime_error("Failed to load " + path);
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

vector<Sample> load_dataset(const string &root_dir, int H, int W, int number_of_labels) {
	vector<Sample> dataset;
	map<string, int> class_to_label;

	int label_counter = 0;

	for (const auto &entry : fs::directory_iterator(root_dir)) {

		if (!entry.is_directory())
			continue;

		string class_name = entry.path().filename().string();
		int label = label_counter++;
		class_to_label[class_name] = label;

		if (label >= number_of_labels)
			break;

		for (const auto &img_entry : fs::directory_iterator(entry)) {
			if (!img_entry.is_regular_file())
				continue;

			string img_path = img_entry.path().string();
			Tensor<float, 3> img_tensor = load_image(img_path, H, W);

			dataset.push_back({img_tensor, label});
		}
	}
	std::random_device rd;
	std::mt19937 g(rd());
	std::shuffle(dataset.begin(), dataset.end(), g);

	return dataset;
}
