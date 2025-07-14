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
			throw invalid_argument("Неправильний вимір матриці");
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
			if (indices[i] >= sizes_[i])
				throw out_of_range("Індекс занадто великий");
			flat_index += strides_[i] * indices[i];
		}
		return data_[flat_index];
	}

	Matrix operator+(Matrix const &other) const {
		if (sizes_ != other.sizes_) {
			throw invalid_argument("Розміри матриць не співпадають");
		}

		Matrix result(sizes_);
		for (size_t i = 0; i < data_.size(); i++)
			result.data_[i] = data_[i] + other.data_[i];
		return result;
	}

	Matrix operator-(Matrix const &other) const {
		if (sizes_ != other.sizes_) {
			throw invalid_argument("Розміри матриць не співпадають");
		}

		Matrix result(sizes_);
		for (size_t i = 0; i < data_.size(); i++)
			result.data_[i] = data_[i] - other.data_[i];
		return result;
	}

  private:
	array<size_t, dim> sizes_;
	array<size_t, dim> strides_;
	vector<T> data_;
};
