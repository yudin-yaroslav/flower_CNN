#include <iostream>
#pragma once

#include <array>
#include <cassert>
#include <cstddef>
#include <initializer_list>
#include <stdexcept>
#include <vector>

using namespace std;

template <size_t dim, typename T> class Matrix {
  public:
	Matrix() = default;

	// TODO: Unite two constructors
	Matrix(const std::array<size_t, dim> &sizes) {
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

	Matrix(initializer_list<size_t> sizes) {
		if (sizes.size() != dim) {
			throw invalid_argument("Wrong matrix dimension");
		}

		copy(sizes.begin(), sizes.end(), sizes_.begin());
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

	template <typename... Idx> T &operator()(Idx... idx) {
		size_t indices[] = {size_t(idx)...};
		size_t flat_index = 0;

		for (size_t i = 0; i < sizeof...(Idx); ++i) {
			flat_index += strides_[i] * indices[i];
		}
		return data_[flat_index];
	}
	template <typename... Idx> const T &operator()(Idx... idx) const { return const_cast<Matrix *>(this)->operator()(idx...); }

	T &operator[](size_t flat_index) { return data_[flat_index]; }

	const T &operator[](size_t flat_index) const { return data_[flat_index]; }

	Matrix slice(const array<size_t, dim * 2> &ranges) const {
		array<size_t, dim> new_sizes;
		for (size_t i = 0; i < dim; ++i) {
			if (ranges[2 * i] > ranges[2 * i + 1] || ranges[2 * i + 1] > sizes_[i])
				throw std::out_of_range("Invalid slice range");
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

	// Elementwise dot product
	T operator*(Matrix const &other) const {
		if (sizes_ != other.sizes_) {
			throw std::invalid_argument("Matrices must have the same size for elementwise dot product");
		}

		T sum = T{};
		for (size_t i = 0; i < sizes_[0]; i++) {
			for (size_t j = 0; j < sizes_[1]; j++) {
				sum += (*this)(i, j) * other(i, j);
			}
		}
		return sum;
	}

	const array<size_t, dim> &get_sizes() const { return sizes_; }

	// TODO: Print matrix

  private:
	array<size_t, dim> sizes_;
	array<size_t, dim> strides_;
	vector<T> data_;
};
