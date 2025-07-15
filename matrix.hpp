#pragma once

#include <array>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace std;

template <size_t dim, typename T> class Matrix {
  public:
	// Constructors
	Matrix() = default;

	Matrix(const array<size_t, dim> &sizes) {
		sizes_ = sizes;
		strides_[dim - 1] = 1;

		for (int i = dim - 2; i >= 0; i--) {
			strides_[i] = strides_[i + 1] * sizes_[i + 1];
		}

		size_t total = 1;
		for (auto s : sizes_) {
			total *= s;
		}

		data_.assign(total, T{});
	}

	template <typename... SizeTs> Matrix(SizeTs... sizes) {
		array<size_t, dim> arr_sizes = {static_cast<size_t>(sizes)...};
		*this = Matrix(arr_sizes);
	}

	// Indexing utils
	array<size_t, dim> index_flat2multi(size_t flat_index) const {
		array<size_t, dim> multi_index;

		size_t remainder = flat_index;
		for (size_t d = 0; d < dim; ++d) {
			multi_index[d] = remainder / strides_[d];
			remainder %= strides_[d];
		}

		return multi_index;
	}

	size_t index_multi2flat(const array<size_t, dim> &multi_index) const {
		size_t flat_index = 0;
		for (size_t i = 0; i < dim; ++i) {
			flat_index += strides_[i] * multi_index[i];
		}
		return flat_index;
	}

	// Slicing
	Matrix slice(const array<size_t, dim * 2> &ranges) const {
		array<size_t, dim> new_sizes;
		for (size_t i = 0; i < dim; ++i) {
			if (ranges[2 * i] > ranges[2 * i + 1] || ranges[2 * i + 1] > sizes_[i])
				throw out_of_range("Invalid slice range");
			new_sizes[i] = ranges[2 * i + 1] - ranges[2 * i];
		}

		Matrix<dim, T> result(new_sizes);

		for (size_t flat_idx = 0; flat_idx < data_.size(); flat_idx++) {
			array<size_t, dim> multi_idx;

			size_t remainder = flat_idx;
			for (size_t d = 0; d < dim; ++d) {
				multi_idx[d] = remainder / strides_[d];
				remainder %= strides_[d];
			}

			bool inside = true;
			array<size_t, dim> out_idx;
			for (size_t d = 0; d < dim; ++d) {
				if (multi_idx[d] < ranges[2 * d] || multi_idx[d] >= ranges[2 * d + 1]) {
					inside = false;
					break;
				}
				out_idx[d] = multi_idx[d] - ranges[2 * d];
			}

			if (inside) {
				size_t flat_out_idx = 0;
				for (size_t d = 0; d < dim; ++d) {
					flat_out_idx += result.strides_[d] * out_idx[d];
				}
				result[flat_out_idx] = data_[flat_idx];
			}
		}
		return result;
	}

	// Indexing operators
	T &operator()(const array<size_t, dim> &multi_index) { return data_[index_multi2flat(multi_index)]; }

	template <typename... Idx, typename = enable_if_t<(sizeof...(Idx) == dim)>> T &operator()(Idx... idx) {
		array<size_t, dim> arr_idx = {static_cast<size_t>(idx)...};
		return (*this)(arr_idx);
	}

	template <typename... Idx, typename = enable_if_t<(sizeof...(Idx) < dim)>, typename = void> Matrix operator()(Idx... idx) {
		array<size_t, 2 * dim> ranges;

		for (size_t k = 0; k < dim; ++k) {
			ranges[2 * k] = 0;
			ranges[2 * k + 1] = sizes_[k];
		}

		size_t indices[] = {static_cast<size_t>(idx)...};
		for (size_t i = 0; i < sizeof...(Idx); ++i) {
			ranges[2 * i] = indices[i];
			ranges[2 * i + 1] = indices[i] + 1;
		}

		return this->slice(ranges);
	}

	T &operator[](size_t flat_index) { return data_[flat_index]; }

	// Other operators
	Matrix operator+(Matrix const &other) const {
		if (sizes_ != other.sizes_) {
			throw invalid_argument("Matrix sizes don't coincide");
		}

		Matrix result(sizes_);
		for (size_t i = 0; i < data_.size(); i++)
			result.data_[i] = data_[i] + other.data_[i];
		return result;
	}

	Matrix operator-(Matrix const &other) const {
		if (sizes_ != other.sizes_) {
			throw invalid_argument("Matrix sizes don't coincide");
		}

		Matrix result(sizes_);
		for (size_t i = 0; i < data_.size(); i++)
			result.data_[i] = data_[i] - other.data_[i];
		return result;
	}

	T operator*(Matrix const &other) const {
		if (sizes_ != other.sizes_) {
			throw invalid_argument("Matrix sizes don't coincide");
		}

		T sum = T{};
		for (size_t i = 0; i < sizes_[0]; i++) {
			for (size_t j = 0; j < sizes_[1]; j++) {
				sum += (*this)(i, j) * other(i, j);
			}
		}
		return sum;
	}

	// Convolution
	static Matrix<2, T> convolute(const Matrix<2, T> &input, const Matrix<2, T> &kernel, size_t stride = 1) {
		size_t input_width = input.get_sizes()[0];
		size_t input_height = input.get_sizes()[1];

		size_t kernel_width = kernel.get_sizes()[0];
		size_t kernel_height = kernel.get_sizes()[1];

		size_t output_width = (input_width - kernel_width) / stride + 1;
		size_t output_height = (input_height - kernel_height) / stride + 1;

		Matrix<2, T> result({output_width, output_height});
		Matrix<2, T> sub_input;

		for (size_t x = 0; x < output_width; x++) {
			for (size_t y = 0; y < output_height; y++) {
				size_t str_x = x * stride;
				size_t str_y = y * stride;

				sub_input = input.slice({str_x, str_x + kernel_width, str_y, str_y + kernel_height});

				result(x, y) = sub_input * kernel;
			}
		}

		return result;
	}

	const array<size_t, dim> &get_sizes() const { return sizes_; }
	const array<size_t, dim> &get_strides() const { return strides_; }
	const vector<T> &get_data() const { return data_; }

	void print() {
		array<size_t, dim> indices{};
		print_recursive(0, indices);
		cout << endl;
	}

  private:
	array<size_t, dim> sizes_;
	array<size_t, dim> strides_;
	vector<T> data_;

	void print_recursive(size_t dim_level, array<size_t, dim> &indices) const {
		if (dim_level == dim - 1) {
			cout << "[ ";
			for (size_t i = 0; i < sizes_[dim_level]; ++i) {
				indices[dim_level] = i;
				cout << data_[index_multi2flat(indices)] << " ";
			}
			cout << "]\n";
		} else {
			cout << "[\n";
			for (size_t i = 0; i < sizes_[dim_level]; ++i) {
				indices[dim_level] = i;
				print_recursive(dim_level + 1, indices);
			}
			cout << "]\n";
		}
	}
};
