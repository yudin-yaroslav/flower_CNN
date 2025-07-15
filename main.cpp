#include "matrix.hpp"

using namespace std;

int main() {
	Matrix<2, int> m1(2, 2);

	m1(0, 0) = 1;
	m1(0, 1) = 2;
	m1(1, 0) = 3;
	m1(1, 1) = 4;
	m1[1] = 4;

	m1.print();
	m1(1).print();

	return 0;
}
