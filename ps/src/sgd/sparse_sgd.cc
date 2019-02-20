#include "sgd/sparse_sgd.h"
#include <gflags/gflags.h>

DEFINE_bool(enable_show_scale_gradient, true, "enable show scale gradient");

namespace paddle {
namespace ps {

void SparseSGDRule::update_value(int row, int col, float** w, float* g2sum, const float** grad, const float* g_scale) {
    static bool show_scale = FLAGS_enable_show_scale_gradient;

    Eigen::Map<const Eigen::MatrixXf> mat_g_scale(g_scale, 1, col);

    local_float_vec().resize(col, 0.0);

    Eigen::Map<Eigen::MatrixXf> mat_add_g2sum(local_float_vec().data(), 1, col);
    mat_add_g2sum = Eigen::MatrixXf::Zero(1, col);

    local_g2sum_vec().resize(col, 0.0);
    memcpy(local_g2sum_vec().data(), g2sum, sizeof(float) * col);

    Eigen::Map<Eigen::MatrixXf> mat_g2sum(local_g2sum_vec().data(), 1, col);

    mat_g2sum = ((mat_g2sum.array() + initial_g2sum).cwiseInverse() * initial_g2sum).cwiseSqrt() * learning_rate;

    local_gradient_vec().resize(col);

    for (auto i = 0u; i < row; ++i) {
        Eigen::Map<Eigen::MatrixXf> mat_w(w[i], 1, col);

        memcpy(local_gradient_vec().data(), grad[i], sizeof(float) * col);

        Eigen::Map<Eigen::MatrixXf> mat_grad(local_gradient_vec().data(), 1, col);

        if (show_scale) {
            mat_grad = mat_grad.cwiseQuotient(mat_g_scale);
        }

        mat_w -= mat_grad.cwiseProduct(mat_g2sum);

        mat_add_g2sum += mat_grad.cwiseProduct(mat_grad);
    }

    Eigen::Map<Eigen::MatrixXf> output_mat_g2sum(g2sum, 1, col);
    output_mat_g2sum += mat_add_g2sum / row;

    bound_value(row, col, w);
}

}
}
