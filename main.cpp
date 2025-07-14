#include "matrix.hpp"
#include <iostream>

int main() {
	Matrix<2, int> m1({2, 2});
	Matrix<2, int> m2({2, 1});
	Matrix<2, int> m3;

	m1(0, 0) = 1;
	m1(0, 1) = 2;
	m1(1, 0) = 3;
	m1(1, 1) = 4;

	m2(0, 0) = 6;
	m2(1, 0) = 10;

	m3 = m1 * m2;

	for (size_t i = 0; i < 2; ++i) {
		for (size_t j = 0; j < 1; ++j)
			std::cout << m3(i, j) << " ";
		std::cout << "\n";
	}

	return 0;
}
