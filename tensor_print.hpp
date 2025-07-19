
#include <iostream>
#include <unsupported/Eigen/CXX11/Tensor>

using namespace Eigen;

template <typename Derived, typename... DimsType>
void PrintTensorNumpy(const Eigen::TensorBase<Derived, Eigen::ReadOnlyAccessors> &input, int dim0, DimsType... dims) {
	constexpr size_t numDimensions = sizeof...(dims) + 1;

	if constexpr (numDimensions > 1) {
		std::cout << "[\n";
		for (int i = 0; i < dim0; ++i) {
			PrintTensorNumpy((static_cast<const Derived &>(input)).chip(i, 0), dims...);
		}
		std::cout << "]\n";
	} else {
		using Scalar = typename Derived::Scalar;
		Eigen::Tensor<Scalar, 1> vec = input.eval();

		std::cout << "[";
		for (int i = 0; i < dim0; ++i) {
			std::cout << vec(i);
			if (i + 1 < dim0)
				std::cout << ", ";
		}
		std::cout << "]\n";
	}
}

template <typename Derived, typename... DimsType>
void PrintMatrixNumpy(const Eigen::MatrixBase<Derived> &input, int dim0, DimsType... dims) {
	constexpr size_t numDimensions = sizeof...(dims);

	if constexpr (numDimensions > 0) {
		std::cout << "[\n";
		for (int i = 0; i < dim0; ++i) {
			PrintMatrixNumpy(input.row(i), dims...);
		}
		std::cout << "]";
	} else {
		std::cout << "[";
		for (int j = 0; j < input.cols(); ++j) {
			std::cout << input(0, j);
			if (j != input.cols() - 1)
				std::cout << ", ";
		}
		std::cout << "]\n";
	}
}
