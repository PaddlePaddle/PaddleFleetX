/***************************************************************************
 * 
 * Copyright (c) 2018 Baidu.com, Inc. All Rights Reserved
 * $Id$ 
 * 
 **************************************************************************/
#include "sgd/sparse_sgd.h"
#include <cmath>
#include <iostream>
#include "proto/ps.pb.h"
#include "gtest/gtest.h" 

using namespace paddle::ps;
using namespace paddle;

TEST(downpour_sparse_sgd_test, test_init_and_update) {
    SparseSGDRule rule;
    SparseSGDRuleParameter param;
    param.set_learning_rate(0.1);
    param.set_initial_g2sum(0.2);
    param.set_initial_range(0.3);
    param.add_weight_bounds(-10.0);
    param.add_weight_bounds(10.0);

    rule.load_config(param);
    ASSERT_FLOAT_EQ(rule.learning_rate, param.learning_rate());
    ASSERT_FLOAT_EQ(rule.initial_g2sum, param.initial_g2sum());
    ASSERT_FLOAT_EQ(rule.initial_range, param.initial_range());
    ASSERT_FLOAT_EQ(rule.min_bound, param.weight_bounds(0));
    ASSERT_FLOAT_EQ(rule.max_bound, param.weight_bounds(1));
   
    // check init_value for zero
    const int item_size = 10;
    float* w = new float[item_size]; 
    float* g2sum = new float[item_size];
    rule.init_value(1, item_size, &w, g2sum, true);

    for (auto i = 0u; i < item_size; ++i) {
        ASSERT_FLOAT_EQ(w[i], 0);
        ASSERT_FLOAT_EQ(g2sum[i], 0);
    }

    // check init_value for random
    rule.init_value(1, item_size, &w, g2sum, false);
    for (auto i = 0u; i < item_size; ++i) {
        ASSERT_TRUE(w[i] >= rule.min_bound && w[i] <= rule.max_bound);
        ASSERT_FLOAT_EQ(g2sum[i], 0);
    }

    // check update_value for one field
    for (auto i = 0u; i < item_size; ++i) {
        w[i] = 0;
        g2sum[i] = 0;
    }
    float* grad = new float[item_size];
    float* g_scale = new float[item_size];
    for (auto i = 0u; i < item_size; ++i) {
        grad[i] = (i + 1) * 1.0;
        g_scale[i] = (i + 1) * 2.0;
    }
    std::vector<float> exp_w;
    std::vector<float> exp_g2sum;
    for (auto i = 0u; i < item_size; ++i) {
        exp_w.push_back(w[i]);
        exp_g2sum.push_back(g2sum[i]);
    }
    for (auto i = 0u; i < item_size; ++i) {
        // use original downpour implementation as standard 
        rule.update_value(1, &exp_w[i], exp_g2sum[i], &grad[i], g_scale[i]);
    }

    const float* ptr_grad = grad;
    const float* ptr_g_scale = g_scale;
    rule.update_value(1, item_size, &w, g2sum, &ptr_grad, ptr_g_scale);

    for (auto i = 0u; i < item_size; ++i) {
        ASSERT_FLOAT_EQ(w[i], exp_w[i]);
        ASSERT_FLOAT_EQ(g2sum[i], exp_g2sum[i]);
    }

}

// check update_value for embedx alike multi-fields
TEST(downpour_sparse_sgd_test, test_update) {
    SparseSGDRule rule;
    SparseSGDRuleParameter param;
    param.set_learning_rate(0.1);
    param.set_initial_g2sum(0.2);
    param.set_initial_range(0.3);
    param.add_weight_bounds(-10.0);
    param.add_weight_bounds(10.0);

    rule.load_config(param);
   
    const int item_size = 10;
    const int field_size = 8;
    float** w = new float* [field_size]; 
    for (auto i = 0u; i < field_size; ++i) {
        w[i] = new float[item_size];
    }
    float* g2sum = new float[item_size];

    // check init_value for random
    rule.init_value(field_size, item_size, w, g2sum, false);
    for (auto i = 0u; i < field_size; ++i) {
        for (auto j = 0u; j < item_size; ++j) {
            ASSERT_TRUE(w[i][j] >= rule.min_bound && w[i][j] <= rule.max_bound);
        }
    }

    // check init_value for zero
    rule.init_value(field_size, item_size, w, g2sum, true);

    std::vector<float> exp_g2sum;
    for (auto i = 0u; i < item_size; ++i) {
        ASSERT_FLOAT_EQ(g2sum[i], 0);

        exp_g2sum.push_back(g2sum[i]);
    }

    for (auto i = 0u; i < field_size; ++i) {
        for (auto j = 0u; j < item_size; ++j) {
            ASSERT_FLOAT_EQ(w[i][j], 0);
        }
    }
    std::vector<std::vector<float>> exp_w;
    for (auto i = 0u; i < item_size; ++i) {
        std::vector<float> exp_w_tmp;
        for (auto j = 0u; j < field_size; ++j) {
            exp_w_tmp.push_back(w[j][i]);
        }
        exp_w.push_back(std::move(exp_w_tmp));
    }

    typedef const float* const_float_ptr;
    const_float_ptr* grad = new const_float_ptr [field_size];
    for (auto i = 0u; i < field_size; ++i) {
        float* ptr = new float[item_size];
        for (auto j = 0u; j < item_size; ++j) {
            ptr[j] = 1.0;

        }
        grad[i] = ptr;
    }

    std::vector<std::vector<float>> grad_vec;
    for (auto i = 0u; i < item_size; ++i) {
        std::vector<float> ptr_vec;
        for (auto j = 0u; j < field_size; ++j) {
            ptr_vec.push_back(grad[j][i]);
        }
        grad_vec.push_back(std::move(ptr_vec));
    }

    // set grad / g_scale == 0.5 for check convinient
    float* g_scale = new float[item_size];
    for (auto i = 0u; i < item_size; ++i) {
        g_scale[i] = 2.0;
    }

    for (auto i = 0u; i < item_size; ++i) {
        rule.update_value(field_size, &exp_w[i][0], exp_g2sum[i], &grad_vec[i][0], g_scale[i]);
    }
    const float* ptr_g_scale = g_scale;
    rule.update_value(field_size, item_size, w, g2sum, grad, ptr_g_scale);

    for (auto i = 0u; i < item_size; ++i) {
        for (auto j = 0u; j < field_size; ++j) {
            ASSERT_FLOAT_EQ(w[j][i], exp_w[i][j]);
        }

        ASSERT_FLOAT_EQ(g2sum[i], exp_g2sum[i]);
    }
}

/* vim: set ts=4 sw=4 sts=4 tw=100 */
