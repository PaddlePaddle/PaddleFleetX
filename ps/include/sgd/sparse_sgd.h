#ifndef BAIDU_BAIDU_PSLIB_SGD_SPARSE_SGD_H
#define BAIDU_BAIDU_PSLIB_SGD_SPARSE_SGD_H
#include <vector>
#include <thread>
#include "glog/logging.h"       // for CHECK
#include "common/local_random.h"    // for local_uniform_real_distribution
#include "Eigen/Dense"
#include "proto/ps.pb.h"

namespace paddle {
namespace ps {

inline std::vector<float>& local_float_vec() {
    thread_local std::vector<float> vec;
    return vec;
}

inline std::vector<float>& local_gradient_vec() {
    thread_local std::vector<float> vec;
    return vec;
}

inline std::vector<float>& local_g2sum_vec() {
    thread_local std::vector<float> vec;
    return vec;
}

inline std::vector<float>& local_score_vec() {
    thread_local std::vector<float> vec;
    return vec;
}

struct SparseSGDRule {
    float learning_rate, initial_g2sum, initial_range;
    float min_bound;
    float max_bound;

    void load_config(const SparseSGDRuleParameter& param) {
        learning_rate = param.learning_rate();
        initial_g2sum = param.initial_g2sum();
        initial_range = param.initial_range();

        if (param.weight_bounds_size() == 0) {
            min_bound = -std::numeric_limits<float>::max();
            max_bound = std::numeric_limits<float>::max();
        } else {
            CHECK(param.weight_bounds_size() >= 2) 
                << "invalid repeated size for weight_bounds:" << param.weight_bounds_size();
            min_bound = param.weight_bounds(0);
            max_bound = param.weight_bounds(1);
        }
    }
    template<class T>
    void init_value(int n, T w[], T& g2sum, bool zero_intialized = false) {
        for (int i = 0; i < n; i++) {
            if (zero_intialized) {
                w[i] = 0.0;
                bound_value(w[i]);
            } else {
                w[i] = (local_uniform_real_distribution<double>()(
                            local_random_engine()) * 2 - 1) * initial_range;
                bound_value(w[i]);
            }
        }

        g2sum = 0;
    }
    void init_value(int row, int col, float** w, float* g2sum, bool zero_intialized = false) {
        if (zero_intialized) {
            for (auto i = 0u; i < row; ++i) {
                Eigen::Map<Eigen::MatrixXf> mat_w(w[i], 1, col);
                mat_w = Eigen::MatrixXf::Zero(1, col);
            }
        } else {
            for (auto i = 0u; i < row; ++i) {
                for (auto j = 0u; j < col; ++j) {
                    w[i][j] = (local_uniform_real_distribution<float>()(
                                local_random_engine()) * 2 - 1) * initial_range;
                }
            }
            bound_value(row, col, w);
        }

        Eigen::Map<Eigen::MatrixXf> mat_g2sum(g2sum, 1, col);
        mat_g2sum = Eigen::MatrixXf::Zero(1, col);
    }

    // naive sgd without g2sum
    template<class T>
    void init_value(int n, T w[], bool zero_intialized = false) {
        for (int i = 0; i < n; i++) {
            if (zero_intialized) {
                w[i] = 0.0;
                bound_value(w[i]);
            } else {
                w[i] = (local_uniform_real_distribution<double>()(
                            local_random_engine()) * 2 - 1) * initial_range;
                bound_value(w[i]);
            }
        }
    }
    void init_value(int row, int col, float** w, bool zero_intialized = false) {
        if (zero_intialized) {
            for (auto i = 0u; i < row; ++i) {
                Eigen::Map<Eigen::MatrixXf> mat_w(w[i], 1, col);
                mat_w = Eigen::MatrixXf::Zero(1, col);
            }
        } else {
            for (auto i = 0u; i < row; ++i) {
                for (auto j = 0u; j < col; ++j) {
                    w[i][j] = (local_uniform_real_distribution<float>()(
                                local_random_engine()) * 2 - 1) * initial_range;
                }
            }
            bound_value(row, col, w);
        }
    }

    template<class T>
    void update_value(int n, T w[], T& g2sum, const T grad[], double g_scale) {
        double add_g2sum = 0;

        for (int i = 0; i < n; i++) {
            double scaled_grad = grad[i] / g_scale;
            w[i] -= learning_rate * scaled_grad * sqrt(initial_g2sum / (initial_g2sum + g2sum));
            bound_value(w[i]);
            add_g2sum += scaled_grad * scaled_grad;
        }

        g2sum += add_g2sum / n;
    }

    void update_value(int row, int col, float** w, float* g2sum, const float** grad, const float* g_scale);

    // naive sgd without g2sum
    template<class T>
    void update_value(int n, T w[], const T grad[]) {
        for (int i = 0; i < n; i++) {
            w[i] -= learning_rate * grad[i];
            bound_value(w[i]);
        }
    }
    void update_value(int row, int col, float** w, const float** grad) {
        for (auto i = 0u; i < row; ++i) {
            Eigen::Map<Eigen::MatrixXf> mat_w(w[i], 1, col);
            Eigen::Map<const Eigen::MatrixXf> mat_grad(grad[i], 1, col);
            mat_w -= learning_rate * mat_grad;
        }

        bound_value(row, col, w);
    }

    template<class T>
    void bound_value(T& w) {
        if (!(w >= min_bound)) {
            w = (T)min_bound;
        } else if (!(w <= max_bound)) {
            w = (T)max_bound;
        }
    }

    void bound_value(int row, int col, float** w) {
        for (auto i = 0u; i < row; ++i) {
            Eigen::Map<Eigen::MatrixXf> mat_w(w[i], 1, col);
            mat_w = mat_w.array().min(max_bound).max(min_bound);
        }
    }
};

}
}
#endif
