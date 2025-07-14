#include "matrix.hpp"
#include <iostream>

int main() {
	Matrix<2, int> m1({3, 4});
	Matrix<2, int> m2({3, 4});
	Matrix<2, int> m3;

	for (size_t i = 0; i < 3; ++i)
		for (size_t j = 0; j < 4; ++j) {
			m1(i, j) = i * 10 + j;
			m2(i, j) = (i + 1) * 100 + j;
		}

	m3 = m1 + m2;

	for (size_t i = 0; i < 3; ++i) {
		for (size_t j = 0; j < 4; ++j)
			std::cout << m3(i, j) << " ";
		std::cout << "\n";
	}

	return 0;
}
