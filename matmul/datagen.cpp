#include <iostream>
#include <fstream>
#include <random>
#include <cstdint>

void generate_matrix(float* matrix, uint64_t rows, uint64_t cols) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    for (uint64_t i = 0; i < rows; ++i) {
        for (uint64_t j = 0; j < cols; ++j) {
            matrix[i * cols + j] = dis(gen);
        }
    }
}

int main() {
    uint64_t n1 = 64;  // Adjust these dimensions as needed
    uint64_t n2 = 32;
    uint64_t n3 = 36864;

    float* matrix1 = new float[n1 * n2];
    float* matrix2 = new float[n2 * n3];

    generate_matrix(matrix1, n1, n2);
    generate_matrix(matrix2, n2, n3);

    std::ofstream file("conf.data", std::ios::out | std::ios::binary);
    if (file.is_open()) {
        file.write(reinterpret_cast<char*>(&n1), sizeof(uint64_t));
        file.write(reinterpret_cast<char*>(&n2), sizeof(uint64_t));
        file.write(reinterpret_cast<char*>(&n3), sizeof(uint64_t));
        file.write(reinterpret_cast<char*>(matrix1), n1 * n2 * sizeof(float));
        file.write(reinterpret_cast<char*>(matrix2), n2 * n3 * sizeof(float));
        file.close();
    } else {
        std::cerr << "Unable to open file for writing.\n";
    }

    delete[] matrix1;
    delete[] matrix2;

    return 0;
}
