#pragma once

#include <cstddef>
#include <iostream>
#include <vector>

using namespace std;

template <size_t dimcount, typename T> class Matrix {
  public:
	using type = std::vector<typename Matrix<dimcount - 1, T>::type>;

	static type add(const type &m1, const type &m2) {
		if (m1.size() != m2.size()) {
			throw invalid_argument("Розміри матриць не співпадають");
		}

		type result(m1.size());
		for (int i = 0; i < m1.size(); i++) {
			result[i] = Matrix<dimcount - 1, T>::add(m1[i], m2[i]);
		}

		return result;
	}
};

template <typename T> class Matrix<0, T> {
  public:
	using type = T;

	static type add(const type &a, const type &b) { return a + b; }
};
