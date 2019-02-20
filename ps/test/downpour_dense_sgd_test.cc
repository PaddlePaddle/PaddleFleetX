/***************************************************************************
 * 
 * Copyright (c) 2018 Baidu.com, Inc. All Rights Reserved
 * $Id$ 
 * 
 **************************************************************************/
#include "sgd/dense_sgd.h"
#include <cmath>
#include <iostream>
#include "proto/ps.pb.h"
#include "gtest/gtest.h" 
#include "common/timer.h"

using namespace paddle::ps;
using namespace paddle;

struct DownpourAdamSGD {
    float w = 0;
    float avg_w = 0;
    float ada_d2sum = 0;
    float ada_g2sum = 0;
    float mom_velocity = 0;

    float learning_rate;
    float ada_decay_rate;
    float ada_epsilon;
    float avg_decay_rate;
    float mom_decay_rate;

    void init() {
        w = 0;
        avg_w = 0;
        ada_g2sum = 0;
        ada_d2sum = 0;
        mom_velocity = 0;
    }
    void downpour_update_adam(float grad) {
        
        ada_d2sum = ada_decay_rate * ada_d2sum + 1;
        ada_g2sum = ada_decay_rate * ada_g2sum + grad * grad;
        float scale = sqrt((1.0 + ada_epsilon) / (ada_g2sum / ada_d2sum + ada_epsilon));
        mom_velocity = mom_decay_rate * mom_velocity - (1 - mom_decay_rate) * grad;
        w += learning_rate * mom_velocity * scale;
        avg_w = avg_decay_rate * avg_w + (1 - avg_decay_rate) * w;
    }
};

TEST(downpour_dense_sgd_test, test_adam_sgd_rule) {
    AdamSGDRule rule;

    DenseSGDRuleParameter param;
    param.set_name("adam");
    param.mutable_adam()->set_learning_rate(0.1);
    param.mutable_adam()->set_avg_decay_rate(0.2);
    param.mutable_adam()->set_ada_decay_rate(0.3);
    param.mutable_adam()->set_ada_epsilon(0.4);
    param.mutable_adam()->set_mom_decay_rate(0.5);

    rule.load_config(param);

    ASSERT_FLOAT_EQ(rule.learning_rate, param.adam().learning_rate());
    ASSERT_FLOAT_EQ(rule.avg_decay_rate, param.adam().avg_decay_rate());
    ASSERT_FLOAT_EQ(rule.ada_decay_rate, param.adam().ada_decay_rate());
    ASSERT_FLOAT_EQ(rule.ada_epsilon, param.adam().ada_epsilon());
    ASSERT_FLOAT_EQ(rule.mom_decay_rate, param.adam().mom_decay_rate());
   
    // check init_value for zero
    //const int item_size = 7865572;
    const int item_size = 7865572 / 50;
    const int field_size = 5;
    float** w = new float*[field_size];
    for (auto i = 0u; i < field_size; ++i) {
        w[i] = new float[item_size];
    }
    rule.init_value(w, item_size);

    for (auto i = 0u; i < field_size; ++i) {
        for (auto j = 0u; j < item_size; ++j) {
            ASSERT_FLOAT_EQ(w[i][j], 0);
        }
    }

    // check merge_value 
    typedef const float * const_float_ptr;
    const_float_ptr* other = new const_float_ptr [field_size];
    for (auto i = 0u; i < field_size; ++i) {
        float* ptr = new float[item_size];
        for (auto j = 0u; j < item_size; ++j) {
            ptr[j] = (i + j) * 1.0;
        }
        other[i] = ptr;
    }

    rule.merge_value(w, other, item_size);
    for (auto i = 0u; i < field_size; ++i) {
        for (auto j = 0u; j < field_size; ++j) {
            ASSERT_FLOAT_EQ(w[i][j], (i + j) * 1.0);
        }
    }

    // check get_weight for pull
    float* pull_value = new float[item_size];
    rule.get_weight(&pull_value, other, item_size);
    for (auto i = 0u; i < item_size; ++i) {
        ASSERT_FLOAT_EQ(pull_value[i], i * 1.0);
    }

    DownpourAdamSGD sgd;
    sgd.learning_rate = rule.learning_rate;
    sgd.ada_decay_rate = rule.ada_decay_rate;
    sgd.ada_epsilon = rule.ada_epsilon;
    sgd.avg_decay_rate = rule.avg_decay_rate;
    sgd.mom_decay_rate = rule.mom_decay_rate;

    std::vector<DownpourAdamSGD> exp_sgd;
    for (auto i = 0u; i < item_size; ++i) {
        sgd.w = w[0][i];
        sgd.avg_w = w[1][i];
        sgd.ada_d2sum = w[2][i];
        sgd.ada_g2sum = w[3][i];
        sgd.mom_velocity = w[4][i];

        sgd.downpour_update_adam(pull_value[i]);

        exp_sgd.push_back(sgd);
    }

    // check update_weight for push
    const_float_ptr ptr = pull_value;
    rule.update_weight(w, &ptr, item_size);
    for (auto i = 0u; i < item_size; ++i) {
        ASSERT_FLOAT_EQ(w[0][i], exp_sgd[i].w);
        ASSERT_FLOAT_EQ(w[1][i], exp_sgd[i].avg_w);
        ASSERT_FLOAT_EQ(w[2][i], exp_sgd[i].ada_d2sum);
        ASSERT_FLOAT_EQ(w[3][i], exp_sgd[i].ada_g2sum);
        ASSERT_FLOAT_EQ(w[4][i], exp_sgd[i].mom_velocity);
    }

    float* push_value = new float[item_size];
    for (auto i = 0u; i < item_size; ++i) {
        push_value[i] = 0;
    }
    const float* push_value_ptr = push_value;
    rule.init_value(w, item_size);
    rule.update_weight(w, &push_value_ptr, item_size);
    //for (auto i = 0u; i < item_size; ++i) {
    //    std::cout << "w[" << i << "]=" << w[0][i] << std::endl;
    //}

    {
        CostTimer timer("test_update_weight");
        for (auto i = 0u; i < 20; ++i) {
            rule.update_weight(w, &push_value_ptr, item_size);
        }
    }

    float grad = 0;
    sgd.init();
    sgd.downpour_update_adam(grad);

    std::cout << "w:" << sgd.w << std::endl;
}

TEST(downpour_dense_sgd_test, test_summary_sgd) {
    SummarySGDRule rule;
    DenseSGDRuleParameter param;
    param.set_name("summary");
    param.mutable_summary()->set_summary_decay_rate(0.1);
    rule.load_config(param);
    ASSERT_FLOAT_EQ(rule.decay_rate, param.summary().summary_decay_rate());

    const int item_size = 10;
    float* w = new float[item_size];
    rule.init_value(&w, item_size);

    for (auto i = 0u; i < item_size; ++i) {
        ASSERT_FLOAT_EQ(w[i], 0);
    }

    // check merge_value
    float* other = new float[item_size];
    for (auto i = 0u; i < item_size; ++i) {
        other[i] = i * 1.0;
    }
    const float* ptr = other;
    rule.merge_value(&w, &ptr, item_size);

    // check get_weight for pull
    float* pull_value = new float[item_size];
    ptr = w;
    rule.get_weight(&pull_value, &ptr, item_size);
    for (auto i = 0u; i < item_size; ++i) {
        ASSERT_FLOAT_EQ(pull_value[i], i * 1.0);
    }

    // check update_weight for push
    ptr = pull_value;
    rule.update_weight(&w, &ptr, item_size);
    for (auto i = 0u; i < item_size; ++i) {
        auto exp_w = i * 1.0 * rule.decay_rate - i * 1.0;
        ASSERT_FLOAT_EQ(w[i], exp_w);
    }
}

TEST(downpour_dense_sgd_test, test_naive_sgd) {
    NaiveSGDRule rule;
    DenseSGDRuleParameter param;
    param.set_name("naive");
    param.mutable_naive()->set_learning_rate(0.1);
    param.mutable_naive()->set_avg_decay_rate(0.2);

    rule.load_config(param);
    ASSERT_FLOAT_EQ(rule.learning_rate, param.naive().learning_rate());

    // check init_value
    const int field_size = 1;
    const int item_size = 786000;
    float** w = new float* [field_size];
    for (auto i = 0u; i < field_size; ++i) {
        w[i] = new float[item_size];
    }
    rule.init_value(w, item_size);
    for (auto i = 0u; i < field_size; ++i) {
        for (auto j = 0u; j < field_size; ++j) {
            ASSERT_FLOAT_EQ(w[i][j], 0);
        }
    }

    // check merge_value
    typedef const float* const_float_ptr;
    const_float_ptr* other = new const_float_ptr[field_size];
    for (auto i = 0u; i < field_size; ++i) {
        float* ptr = new float[item_size];
        for (auto j = 0u; j < item_size; ++j) {
            ptr[j] = (i + j) * 1.0;
        }
        other[i] = ptr;
    }
    rule.merge_value(w, other, item_size);
    for (auto i = 0u; i < field_size; ++i) {
        for (auto j = 0u; j < item_size; ++j) {
            ASSERT_FLOAT_EQ(w[i][j], (i + j) * 1.0);
        }
    }

    // check get_weight for pull
    float* pull_value = new float[item_size];
    rule.get_weight(&pull_value, other, item_size);
    for (auto i = 0u; i < item_size; ++i) {
        ASSERT_FLOAT_EQ(pull_value[i], i * 1.0);
    }

    const float* ptr = pull_value;
    // check update_weight for push
    rule.update_weight(w, &ptr, item_size);
    for (auto i = 0u; i < field_size; ++i) {
        auto exp_w = (1 - rule.learning_rate) * i;
        ASSERT_FLOAT_EQ(w[0][i], exp_w);
    }

    float* push_value = new float[item_size];
    for (auto i = 0u; i < item_size; ++i) {
        push_value[i] = 0;
    }
    const float* push_value_ptr = push_value;
    rule.init_value(w, item_size);
    {
        CostTimer timer("update_weight_naive_timecost");
        for (auto i = 0; i < 20; ++i) {
            rule.update_weight(w, &push_value_ptr, item_size);
        }
    }
}

TEST(downpour_dense_sgd_test, test_adam_save_load) {
    AdamSGDRule rule;
    const int field_size = 5;
    float* v = new float[field_size];
    for (auto i = 0u; i < field_size; ++i) {
        v[i] = i;
    }
    auto str = rule.to_string(v);
    EXPECT_STREQ(str.c_str(), "0 1 2 3 4");

    std::string l_str = "1 2 3 4 5";
    ASSERT_TRUE(rule.from_string(l_str, v) == 0);
    for (auto i = 0u; i < field_size; ++i) {
        ASSERT_FLOAT_EQ(v[i], i + 1);
    }
}

TEST(downpour_dense_sgd_test, test_summary_sgd_save_load) {
    SummarySGDRule rule;
    float v = 1;
    auto str = rule.to_string(&v);

    ASSERT_TRUE(strncmp(str.c_str(), "1", 1) == 0);

    str = "2";
    ASSERT_TRUE(rule.from_string(str, &v) == 0);
    ASSERT_FLOAT_EQ(v, 2);
}


/* vim: set ts=4 sw=4 sts=4 tw=100 */
