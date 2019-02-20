#include "sgd/dense_sgd.h"
#include <cmath>
#include "Eigen/Dense"
#include "common/timer.h"
#include "common/common.h"
namespace paddle {
namespace ps {

void AdamSGDRule::load_config(const DenseSGDRuleParameter& param) {
    DenseSGDRule::load_config(param);
    learning_rate = param.adam().learning_rate();
    avg_decay_rate = param.adam().avg_decay_rate();
    ada_decay_rate = param.adam().ada_decay_rate();
    ada_epsilon = param.adam().ada_epsilon();
    mom_decay_rate = param.adam().mom_decay_rate();
}

void AdamSGDRule::init_value(float** value, size_t num) {
    for (auto i = 0u; i < AdamSGDValue::dim(); ++i) {
        Eigen::Map<Eigen::MatrixXf> mat(value[i], 1, num);
        mat = Eigen::MatrixXf::Zero(1, num);
    }
}

void AdamSGDRule::merge_value(float** value, const float** from, size_t num) {
    for (auto i = 0u; i < AdamSGDValue::dim(); ++i) {
        Eigen::Map<Eigen::MatrixXf> mat(value[i], 1, num);
        Eigen::Map<const Eigen::MatrixXf> mat_from(from[i], 1, num);
        mat += mat_from;
    }
}

void AdamSGDRule::get_weight(float** value, const float** from, size_t num) {
    memcpy(value[0], from[AdamSGDValue::w_index()], sizeof(float) * num);
}

void AdamSGDRule::set_weight(const float** w, float** value, size_t num) {
    memcpy(value[AdamSGDValue::w_index()], w[0], sizeof(float) * num);
}

float AdamSGDRule::get_avg_weight(float* value) {
    return AdamSGDValue::avg_w(value);
}

void AdamSGDRule::set_avg_weight(float avg_w, float* value) {
    AdamSGDValue::avg_w(value) = avg_w;
}

void AdamSGDRule::update_weight(float** value, const float** grad, size_t num) {
    Eigen::Map<Eigen::MatrixXf> mat_ada_g2sum(value[AdamSGDValue::ada_g2sum_index()], 1, num);
    Eigen::Map<Eigen::MatrixXf> mat_ada_d2sum(value[AdamSGDValue::ada_d2sum_index()], 1, num);
    Eigen::Map<Eigen::MatrixXf> mat_mom_velocity(value[AdamSGDValue::mom_velocity_index()], 1, num);
    Eigen::Map<Eigen::MatrixXf> mat_w(value[AdamSGDValue::w_index()], 1, num);
    Eigen::Map<Eigen::MatrixXf> mat_avg_w(value[AdamSGDValue::avg_w_index()], 1, num);

    Eigen::Map<const Eigen::MatrixXf> mat_grad(grad[0], 1, num);

    mat_ada_d2sum = (mat_ada_d2sum * ada_decay_rate).array() + 1;
    mat_ada_g2sum = (mat_ada_g2sum * ada_decay_rate) + mat_grad.cwiseProduct(mat_grad);

    adam_local_scale().resize(num);
    Eigen::Map<Eigen::MatrixXf> scale(adam_local_scale().data(), 1, num);
    memcpy(adam_local_scale().data(), mat_ada_d2sum.data(), sizeof(float) * num);
    scale = scale.array() * ada_epsilon;
    scale = (mat_ada_d2sum + scale).cwiseQuotient(mat_ada_g2sum + scale);
    scale = scale.cwiseSqrt();
    mat_mom_velocity = (mat_mom_velocity + mat_grad) * mom_decay_rate - mat_grad;

    mat_w += learning_rate * mat_mom_velocity.cwiseProduct(scale);
    mat_avg_w = (mat_avg_w - mat_w) * avg_decay_rate + mat_w;
}

const std::string& AdamSGDRule::get_sgd_type() const {
    const static std::string sgd_type("adam");
    return sgd_type;
}

const std::string& AdamSGDRule::get_sgd_class() const {
    const static std::string sgd_class("dnn");
    return sgd_class;
}

std::string AdamSGDRule::to_string(const float* value) {
    return AdamSGDValue::to_string(value);
}

int AdamSGDRule::from_string(const std::string& str, float* v) {
    auto ret = str_to_float(str.data(), v);
    CHECK(ret == dim()) << "expect[" << dim() << "] real[" << ret << "]";
    return 0;
}

void NaiveSGDRule::load_config(const DenseSGDRuleParameter& param) {
    DenseSGDRule::load_config(param);
    learning_rate = param.naive().learning_rate();
}

void NaiveSGDRule::init_value(float** value, size_t num) {
    Eigen::Map<Eigen::MatrixXf> mat_w(value[0], 1, num);
    mat_w = Eigen::MatrixXf::Zero(1, num);
}

void NaiveSGDRule::merge_value(float** value, const float** from, size_t num) {
    Eigen::Map<Eigen::MatrixXf> mat_w(value[0], 1, num);

    Eigen::Map<const Eigen::MatrixXf> mat_w_f(from[0], 1, num);

    mat_w += mat_w_f;
}

void NaiveSGDRule::get_weight(float** value, const float** from, size_t num) {
    memcpy(value[0], from[0], sizeof(float) * num);
}

void NaiveSGDRule::set_weight(const float** w, float** value, size_t num) {
    memcpy(value[0], w[0], sizeof(float) * num);
}

float NaiveSGDRule::get_avg_weight(float* value) {
    return NaiveSGDValue::w(value);
}

void NaiveSGDRule::set_avg_weight(float avg_w, float* value) {
    NaiveSGDValue::w(value) = avg_w;
}

void NaiveSGDRule::update_weight(float** value, const float** grad, size_t num) {
    Eigen::Map<Eigen::MatrixXf> mat_w(value[0], 1, num);
    Eigen::Map<const Eigen::MatrixXf> mat_grad(grad[0], 1, num);
    mat_w -= learning_rate * mat_grad;
}

const std::string& NaiveSGDRule::get_sgd_type() const {
    const static std::string sgd_type("naive");
    return sgd_type;
}

const std::string& NaiveSGDRule::get_sgd_class() const {
    const static std::string sgd_class("dnn");
    return sgd_class;
}

std::string NaiveSGDRule::to_string(const float* value) {
    return NaiveSGDValue::to_string(value);
}

int NaiveSGDRule::from_string(const std::string& str, float* v) {
    CHECK(str_to_float(str.data(), v) == dim());
    return 0;
}

void SummarySGDRule::load_config(const DenseSGDRuleParameter& param) {
    DenseSGDRule::load_config(param);
    decay_rate = param.summary().summary_decay_rate();
}

void SummarySGDRule::init_value(float** value, size_t num) {
    Eigen::Map<Eigen::MatrixXf> mat(value[0], 1, num);
    mat = Eigen::MatrixXf::Zero(1, num);
}

void SummarySGDRule::merge_value(float** value, const float** from, size_t num) {
    Eigen::Map<Eigen::MatrixXf> mat(value[0], 1, num);
    Eigen::Map<const Eigen::MatrixXf> mat_f(from[0], 1, num);

    mat += mat_f;
}

void SummarySGDRule::get_weight(float** value, const float** from, size_t num) {
    memcpy(value[0], from[0], sizeof(float) * num);
}

void SummarySGDRule::set_weight(const float** w, float** value, size_t num) {
    memcpy(value[0], w[0], sizeof(float) * num);
}

float SummarySGDRule::get_avg_weight(float* value) {
    return SummarySGDValue::w(value);
}

void SummarySGDRule::set_avg_weight(float avg_w, float* value) {
    SummarySGDValue::w(value) = avg_w;
}

void SummarySGDRule::update_weight(float** value, const float** grad, size_t num) {
    Eigen::Map<Eigen::MatrixXf> mat_w(value[0], 1, num);
    Eigen::Map<const Eigen::MatrixXf> mat_grad(grad[0], 1, num);
    mat_w = mat_w * decay_rate + mat_grad;
}

const std::string& SummarySGDRule::get_sgd_type() const {
    const static std::string sgd_type("summary");
    return sgd_type;
}

const std::string& SummarySGDRule::get_sgd_class() const {
    const static std::string sgd_class("summary");
    return sgd_class;
}

std::string SummarySGDRule::to_string(const float* value) {
    return SummarySGDValue::to_string(value);
}

int SummarySGDRule::from_string(const std::string& str, float* v) {
    CHECK(str_to_float(str.data(), v) == dim());
    return 0;
}

void MovingAverageRule::load_config(const DenseSGDRuleParameter& param) {
    DenseSGDRule::load_config(param);
    decay_rate = param.moving_average().momentum();
}

void MovingAverageRule::init_value(float** value, size_t num) {
    Eigen::Map<Eigen::MatrixXf> mat(value[0], 1, num);
    mat = Eigen::MatrixXf::Zero(1, num);
}

void MovingAverageRule::merge_value(float** value, const float** from, size_t num) {
    Eigen::Map<Eigen::MatrixXf> mat(value[0], 1, num);
    Eigen::Map<const Eigen::MatrixXf> mat_f(from[0], 1, num);

    mat += mat_f;
}

void MovingAverageRule::get_weight(float** value, const float** from, size_t num) {
    memcpy(value[0], from[0], sizeof(float) * num);
}

void MovingAverageRule::set_weight(const float** w, float** value, size_t num) {
    memcpy(value[0], w[0], sizeof(float) * num);
}

float MovingAverageRule::get_avg_weight(float* value) {
    return value[0];
}

void MovingAverageRule::set_avg_weight(float avg_w, float* value) {
    value[0] = avg_w;
}

void MovingAverageRule::update_weight(float** value, const float** grad, size_t num) {
    Eigen::Map<Eigen::MatrixXf> mat_w(value[0], 1, num);
    Eigen::Map<const Eigen::MatrixXf> mat_grad(grad[0], 1, num);
    mat_w = mat_w * decay_rate + mat_grad * (1 - decay_rate);
}

const std::string& MovingAverageRule::get_sgd_type() const {
    const static std::string sgd_type("moving_average");
    return sgd_type;
}

const std::string& MovingAverageRule::get_sgd_class() const {
    const static std::string sgd_class("moving_average");
    return sgd_class;
}

std::string MovingAverageRule::to_string(const float* value) {
    return std::to_string(value[0]);
}

int MovingAverageRule::from_string(const std::string& str, float* v) {
    CHECK(str_to_float(str.data(), v) == dim());
    return 0;
}

}
}
