/***************************************************************************
 * 
 * Copyright (c) 2018 Baidu.com, Inc. All Rights Reserved
 * $Id$ 
 * 
 **************************************************************************/
#include "common/common.h"
#include "common/matrix.h"
#include "table/downpour_sparse_table.h"
#include "gtest/gtest.h" 

using namespace paddle::ps;

TEST(downpour_sparse_table, Register) {
    TableManager::instance().initialize();
    
    //Create table by class_name
    std::string table_class_name("DownpourSparseTable_Float15"); 
    auto* downpour_sparse_table = TableRegisterer::CreateInstanceByName(table_class_name);
    ASSERT_TRUE(downpour_sparse_table != NULL);
   
    table_class_name = "DownpourSparseTable_Float128"; 
    downpour_sparse_table = TableRegisterer::CreateInstanceByName(table_class_name);
    ASSERT_TRUE(downpour_sparse_table != NULL);
 
    //Initialize table
    ::paddle::TableParameter table_config;
    ::paddle::TableAccessorParameter accessor_config;
    accessor_config.set_accessor_class("DownpourFeatureValueAccessor");
    table_config.mutable_accessor()->CopyFrom(accessor_config);
    ::paddle::FsClientParameter fs_client_config;

    ASSERT_TRUE(downpour_sparse_table->initialize(table_config, fs_client_config) == 0);

    //Check table & accessor
    auto* accessor = downpour_sparse_table->value_accesor().get();
    ASSERT_TRUE(accessor != NULL);
    ASSERT_EQ(accessor->dim(), 7);

}

TEST(downpour_sparse_table, array_eigen_calculate) {
    Eigen::MatrixXf matrix;
    matrix.setZero(1300000, 6);
    matrix(0, 0) = 1;
    matrix(0, 1) = 2;
    matrix(0, 2) = 3;
    matrix(1, 0) = 7;
    matrix(1, 1) = 8;
    matrix(1, 2) = 9;
    float* result_array = new float[matrix.size()];
    std::cout << "row_ptr:" << result_array << "\n";
    {
        paddle::ps::CostTimer timer("copy by for");
        size_t row = matrix.rows();
        size_t col = matrix.cols();
        float* matrix_data = matrix.data();
        for (size_t i = 0; i < row; ++i) {
            float* result_ptr = result_array + i * col; 
            for (size_t j = 0; j < col; ++j) {
                result_ptr[j] = matrix_data[ j * row + i];
            }
        }
    }
    std::cout << result_array[0] << " " << result_array[1] << " " << result_array[2] << "\n";
    std::cout << result_array[6] << " " << result_array[7] << " " << result_array[8] << "\n";
    
    {
        paddle::ps::CostTimer timer("copyArray by transpose_inplace");
        matrix.transposeInPlace();
        memcpy(result_array, matrix.data(), matrix.size() * sizeof(float));
    }
    matrix.transposeInPlace();  //恢复矩阵
    std::cout << result_array[0] << " " << result_array[1] << " " << result_array[2] << "\n";
    
    ///////////////////////////////////////////////////////////////////////////////////
    
    {
        paddle::ps::CostTimer timer("copyFromArray by memcpy");
        size_t row = matrix.rows();
        size_t col = matrix.cols();
        matrix.resize(col, row);
        memcpy(matrix.data(), result_array, matrix.size() * sizeof(float));
        matrix.transposeInPlace();
    }
    std::cout << matrix(0,0) << " " << matrix(0, 1) << " " << matrix(0, 2) << "\n";
    {
        paddle::ps::CostTimer timer("copyFromArray by for");
        size_t row = matrix.rows();
        size_t col = matrix.cols();
        float* matrix_data = matrix.data();
        for (size_t i = 0; i < row; ++i) {
            float* result_data = result_array + i * col;
            for (size_t j = 0; j < col; ++j) {
                matrix_data[j * row + i] = result_data[j];
            }
        }
    }
    std::cout << matrix(0,0) << " " << matrix(0, 1) << " " << matrix(0, 2) << "\n";
    {
        paddle::ps::CostTimer timer("matrix resize");
        Eigen::MatrixXf mat;
        mat.resize(1300000, 6);
    }
    {
        paddle::ps::CostTimer timer("matrix set_zero");
        Eigen::MatrixXf mat;
        mat.setZero(1300000, 6);
    }
}

TEST(downpour_sparse_table, matrix_eigen_calculate) {
    size_t rown = 100000;
    size_t coln = 50;
    float* matrix[rown];
    for (size_t i = 0; i < rown; ++i) {
        matrix[i] = new float[coln];
        for (size_t j = 0; j < coln; ++j) {
            matrix[i][j] = i * coln + j;
        }
    }
    
    Eigen::MatrixXf eigen;
    eigen.resize(rown, coln);

    {
        CostTimer timer("CopyMatrix by for");
        paddle::ps::copy_matrix_to_eigen((const float**)matrix, eigen);
    }
    {
        CostTimer timer("CopyMatrix by new for");
        size_t row = eigen.rows();
        size_t col = eigen.cols();
        float* eigen_data = eigen.data();
        for (size_t i = 0; i < row; ++i) {
            float* matrix_data =  matrix[i];
            for (size_t j = 0; j < col; ++j) {
                eigen_data[j * row + i] = matrix_data[j];
            }
        }
    }
    std::cout << "eigen size:" << eigen.rows() << "x" << eigen.cols() << "\n";
    std::cout << eigen(0, 0) << " " <<  eigen(0, 1) << "\n" << eigen(1, 0) << " " << eigen(1, 1) << "\n";
    {
        CostTimer timer("CopyMatrix by for row");
        eigen.resize(coln, rown);
        for (size_t i = 0; i < rown; ++i) {
            memcpy(eigen.data() + i * coln, matrix[i], coln * sizeof(float));
        }
        eigen.transposeInPlace();
    }
    std::cout << "eigen size:" << eigen.rows() << "x" << eigen.cols() << "\n";
    std::cout << eigen(0, 0) << " " <<  eigen(0, 1) << "\n" << eigen(1, 0) << " " << eigen(1, 1) << "\n";
    
    {
        CostTimer timer("CopyEigen by for row");
        auto arr_data = eigen.array();
        float* data = arr_data.data();
        for (size_t i = 0; i < rown; ++i) {
            memcpy(matrix[i], arr_data.data() + i * coln, coln * sizeof(float));
        }
        std::cout << data[0] << " " << data[1] << "\n";
    }
    std::cout << matrix[0][0] << " " <<  matrix[0][1] << "\n" << matrix[1][0] << " " << matrix[1][1] << "\n";
    {
        CostTimer timer("CopyEigen by for");
        paddle::ps::copy_eigen_to_matrix(eigen, matrix);
    }
    std::cout << matrix[0][0] << " " <<  matrix[0][1] << "\n" << matrix[1][0] << " " << matrix[1][1] << "\n";
}


















/* vim: set ts=4 sw=4 sts=4 tw=100 */
