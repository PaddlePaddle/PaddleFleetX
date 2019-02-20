#pragma once

#include "Eigen/Dense"

namespace paddle {
namespace ps {

inline void make_matrix_with_eigen(size_t row, size_t col,
    Eigen::MatrixXf& eigen_matrix, float** result_matrix) {
    eigen_matrix.resize(row, col);
    for (size_t i = 0; i < col; ++i) {
        result_matrix[i] = eigen_matrix.data() + i * row;
    }
}

inline void copy_matrix_to_eigen(const float** matrix, Eigen::MatrixXf& eigen) {
    size_t row = eigen.rows();
    size_t col = eigen.cols();
    float* eigen_data = const_cast<float*>(eigen.data());
    for (size_t i = 0; i < row; ++i) {
        const float* matrix_array =  matrix[i];
        for (size_t j = 0; j < col; ++j) {
            eigen_data[j * row + i] = matrix_array[j];
        }
    }
}

inline void copy_eigen_to_matrix(const Eigen::MatrixXf& eigen, float** result_matrix) {
    size_t row = eigen.rows();
    size_t col = eigen.cols();
    const float* eigen_data = eigen.data();
    for (size_t i = 0; i < row; ++i) {
        float* matrix_array =  result_matrix[i];
        for (size_t j = 0; j < col; ++j) {
            matrix_array[j] = eigen_data[j * row + i];
        }
    }
}

inline void copy_eigen_to_array(const Eigen::MatrixXf& matrix, float* result_array) {
    size_t row = matrix.rows();
    size_t col = matrix.cols();
    float* matrix_data = const_cast<float*>(matrix.data());
    for (size_t i = 0; i < row; ++i) {
        float* array_ptr = result_array + i * col; 
        for (size_t j = 0; j < col; ++j) {
            array_ptr[j] = matrix_data[j * row + i];
        }
    }
}

inline void copy_array_to_eigen(const float* array_data, Eigen::MatrixXf& matrix) {
    size_t row = matrix.rows();
    size_t col = matrix.cols();
    float* matrix_data = const_cast<float*>(matrix.data());
    for (size_t i = 0; i < row; ++i) {
        const float* array_ptr = array_data + i * col;
        for (size_t j = 0; j < col; ++j) {
            matrix_data[j * row + i] = array_ptr[j];
        }
    }
}

inline std::string matrix_to_string(
    const Eigen::MatrixXf& matrix, const uint64_t* keys = NULL) {
    size_t row = matrix.rows();
    size_t col = matrix.cols();
    std::stringstream ssm;
    for (size_t i = 0; i < row; ++i) {
        if (keys) {
            ssm << keys[i] << ": ";
        }
        for (size_t j = 0; j < col; ++j) {
            ssm << matrix(i, j) << " ";
        }
        ssm << "\n";
    }
    return ssm.str();
}
}
}
