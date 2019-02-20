/***************************************************************************
 * 
 * Copyright (c) 2018 Baidu.com, Inc. All Rights Reserved
 * $Id$ 
 * 
 **************************************************************************/
#include "table/downpour_accessor.h"
#include <cmath>
#include <iostream>
#include "proto/ps.pb.h"
#include "gtest/gtest.h" 

using namespace paddle::ps;
using namespace paddle;

TableAccessorParameter gen_param() {
    TableAccessorParameter param;
    param.set_accessor_class("DownpourFeatureValueAccessor");
    param.set_fea_dim(11);
    param.set_embedx_dim(8);
    param.mutable_downpour_accessor_param()->set_nonclk_coeff(0.2);
    param.mutable_downpour_accessor_param()->set_click_coeff(1);
    param.mutable_downpour_accessor_param()->set_base_threshold(0.5);
    param.mutable_downpour_accessor_param()->set_delta_threshold(0.2);
    param.mutable_downpour_accessor_param()->set_delta_keep_days(16);
    param.mutable_downpour_accessor_param()->set_show_click_decay_rate(0.99);

    param.mutable_sparse_sgd_param()->set_learning_rate(0.1);
    param.mutable_sparse_sgd_param()->set_initial_g2sum(0.2);
    param.mutable_sparse_sgd_param()->set_initial_range(0.3);
    param.mutable_sparse_sgd_param()->add_weight_bounds(-10.0);
    param.mutable_sparse_sgd_param()->add_weight_bounds(10.0);

    return std::move(param);
}

TEST(downpour_feature_value_accessor_test, test_shrink) {
    TableAccessorParameter parameter = gen_param();
    DownpourFeatureValueAccessor* acc = new DownpourFeatureValueAccessor();
    ASSERT_EQ(acc->configure(parameter), 0);
    ASSERT_EQ(acc->initialize(), 0);
    
    float* value = new float[acc->dim()];
    for (auto i = 0u; i < acc->dim(); ++i) {
        value[i] = i * 1.0;
    }
    ASSERT_TRUE(!acc->shrink(value));

    // set unseen_days too long
    value[0] = 1000;
    // set delta score too small
    value[1] = 0.001;
    ASSERT_TRUE(acc->shrink(value));
}

TEST(downpour_feature_value_accessor_test, test_save) {
    TableAccessorParameter parameter = gen_param();
    DownpourFeatureValueAccessor* acc = new DownpourFeatureValueAccessor();
    ASSERT_EQ(acc->configure(parameter), 0);
    ASSERT_EQ(acc->initialize(), 0);

    float* value = new float[acc->dim()];
    for (auto i = 0u; i < acc->dim(); ++i) {
        value[i] = i * 1.0;
    }

    // save all feature
    ASSERT_TRUE(acc->save(value, 0));

    // save delta feature
    ASSERT_TRUE(acc->save(value, 1));

    // save base feature with time decay
    ASSERT_TRUE(acc->save(value, 2));

    ASSERT_FLOAT_EQ(value[2], 0.99 * 2);
    ASSERT_FLOAT_EQ(value[3], 0.99 * 3);
}

TEST(downpour_feature_value_accessor_test, test_create) {
    TableAccessorParameter parameter = gen_param();
    DownpourFeatureValueAccessor* acc = new DownpourFeatureValueAccessor();
    ASSERT_EQ(acc->configure(parameter), 0);
    ASSERT_EQ(acc->initialize(), 0);

    const int field_size = 7 + 8;
    const int item_size = 100;

    float** value = new float* [field_size];
    for (auto i = 0u; i < field_size; ++i) {
        value[i] = new float[item_size];
    }
    ASSERT_EQ(acc->create(value, item_size), 0);

    for (auto i = 0u; i < 4; ++i) {
        for (auto j = 0u; j < item_size; ++j) {
            ASSERT_FLOAT_EQ(value[i][j], 0);
        }
    }
}

TEST(downpour_feature_value_accessor_test, test_update) {
    TableAccessorParameter parameter = gen_param();
    DownpourFeatureValueAccessor* acc = new DownpourFeatureValueAccessor();
    ASSERT_EQ(acc->configure(parameter), 0);
    ASSERT_EQ(acc->initialize(), 0);

    const int field_size = 7 + 8;
    const int item_size = 100;

    float** value = new float* [field_size];
    for (auto i = 0u; i < field_size; ++i) {
        value[i] = new float[item_size];

        for (auto j = 0u; j < item_size; ++j) {
            value[i][j] = 0;
        }
    }

    typedef const float* const_float_ptr;
    const_float_ptr* grad = new const_float_ptr [acc->update_dim()];
    for (auto i = 0u; i < acc->update_dim(); ++i) {
        float* p = new float[item_size];
        for (auto j = 0u; j < item_size; ++j) {
            p[j] = j + 1;
        }
        grad[i] = p;
    }

    struct DownpourSparseValueTest {
        float unseen_days;
        float delta_score;
        float show;
        float click;
        float embed_w;
        float embed_g2sum;
        float embedx_g2sum;
        std::vector<float> embedx_w;

        void to_array(float* ptr, size_t dim) {
            ptr[0] = unseen_days;
            ptr[1] = delta_score;
            ptr[2] = show;
            ptr[3] = click;
            ptr[4] = embed_w;
            ptr[5] = embed_g2sum;
            ptr[6] = embedx_g2sum;
            for (auto j = 0u; j < dim; ++j) {
                ptr[7 + j] = embedx_w[j];
            }
        }
    };
    struct DownpourSparsePushValueTest {
        float show;
        float click;
        float embed_g;
        std::vector<float> embedx_g;
    };
    std::vector<float*> exp_value;
    for (auto i = 0u; i < item_size; ++i) {
        DownpourSparseValueTest v;
        v.unseen_days = value[0][i];
        v.delta_score = value[1][i];
        v.show = value[2][i];
        v.click = value[3][i];
        v.embed_w = value[4][i];
        v.embed_g2sum = value[5][i];
        v.embedx_g2sum = value[6][i];
        for (auto j = 0; j < parameter.embedx_dim(); ++j) {
            v.embedx_w.push_back(value[j + 7][i]);
        }

        DownpourSparsePushValueTest push_v;
        push_v.show = grad[0][i];
        push_v.click = grad[1][i];
        push_v.embed_g = grad[2][i];
        for (auto j = 0; j < parameter.embedx_dim(); ++j) {
            push_v.embedx_g.push_back(grad[3 + j][i]);
        }

        v.unseen_days = 0;
        v.show += push_v.show;
        v.click += push_v.click;
        v.delta_score += acc->show_click_score(push_v.show, push_v.click);

        acc->_embed_sgd_rule.update_value(1, &v.embed_w, v.embed_g2sum, &push_v.embed_g, push_v.show);
        acc->_embedx_sgd_rule.update_value(parameter.embedx_dim(), &v.embedx_w[0], v.embedx_g2sum, &push_v.embedx_g[0], push_v.show);

        float* ptr = new float[acc->dim()];
        v.to_array(ptr, parameter.embedx_dim());
        exp_value.push_back(ptr);
    }
    acc->update(value, grad, item_size);

    for (auto i = 0u; i < item_size; ++i) {
        for (auto j = 0u; j < acc->dim(); ++j) {

            ASSERT_FLOAT_EQ(value[j][i], exp_value[i][j]);
        }
    }
}

TEST(downpour_feature_value_accessor_test, test_show_click_score) {
    TableAccessorParameter parameter = gen_param();
    DownpourFeatureValueAccessor* acc = new DownpourFeatureValueAccessor();
    ASSERT_EQ(acc->configure(parameter), 0);
    ASSERT_EQ(acc->initialize(), 0);

    float show = 10;
    float click = 6;
    ASSERT_FLOAT_EQ(acc->show_click_score(show, click), 6.8);
}

TEST(downpour_feature_value_accessor_test, test_string_related) {
    TableAccessorParameter parameter = gen_param();
    DownpourFeatureValueAccessor* acc = new DownpourFeatureValueAccessor();
    ASSERT_EQ(acc->configure(parameter), 0);
    ASSERT_EQ(acc->initialize(), 0);

    const int field_size = 15;
    float* value = new float[field_size];
    for (auto i = 0u; i < field_size; ++i) {
        value[i] = i;
    }

    auto str = acc->parse_to_string(value, 0);

    std::cout << str << std::endl;

    str = "1 2 3 4 5 6 7";
    ASSERT_TRUE(acc->parse_from_string(str, value) == 0);
    for (auto i = 7; i < 15; ++i) {
        ASSERT_FLOAT_EQ(value[i], 0);
    }
}

/* vim: set ts=4 sw=4 sts=4 tw=100 */
