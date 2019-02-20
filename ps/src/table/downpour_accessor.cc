#include "table/downpour_accessor.h"
#include <gflags/gflags.h>
#include "ps_instance.h"
#include "glog/logging.h"
#include "sgd/dense_sgd.h"
#include "sgd/dense_sgd_factory.h"
#include "common/ps_string.h"
#include "common/common.h"

DEFINE_bool(pslib_shrink_embedx_when_pull, true, "whether shrink embedx when pulled");

namespace paddle {
namespace ps {

size_t DownpourFeatureValueAccessor::dim() {
    auto embedx_dim = _config.embedx_dim();
    return DownpourFeatureValue::dim(embedx_dim);
}

size_t DownpourFeatureValueAccessor::dim_size(size_t dim) {
    auto embedx_dim = _config.embedx_dim();
    return DownpourFeatureValue::dim_size(dim, embedx_dim);
}

size_t DownpourFeatureValueAccessor::size() {
    auto embedx_dim = _config.embedx_dim();
    return DownpourFeatureValue::size(embedx_dim);
}

size_t DownpourFeatureValueAccessor::mf_size() {
    return (_config.embedx_dim() + 1) * sizeof(float);//embedx embedx_g2sum
}

// pull value
size_t DownpourFeatureValueAccessor::select_dim() {
    auto embedx_dim = _config.embedx_dim();
    return 3 + embedx_dim;
}

size_t DownpourFeatureValueAccessor::select_dim_size(size_t dim) {
    return sizeof(float);
}

size_t DownpourFeatureValueAccessor::select_size() {
    return select_dim() * sizeof(float);
}

// push value
size_t DownpourFeatureValueAccessor::update_dim() {
    auto embedx_dim = _config.embedx_dim();
    return 3 + embedx_dim;
}

size_t DownpourFeatureValueAccessor::update_dim_size(size_t dim) {
    return sizeof(float);
}

size_t DownpourFeatureValueAccessor::update_size() {
    return update_dim() * sizeof(float);
}

bool DownpourFeatureValueAccessor::shrink(float* value) {
    //auto base_threshold = _config.downpour_accessor_param().base_threshold();
    //auto delta_threshold = _config.downpour_accessor_param().delta_threshold();
    //auto delta_keep_days = _config.downpour_accessor_param().delta_keep_days();
    //auto delete_threshold = _config.downpour_accessor_param().delete_threshold();
    auto base_threshold = _config.downpour_accessor_param().base_threshold();
    auto delta_threshold = _config.downpour_accessor_param().delta_threshold();
    auto delta_keep_days = _config.downpour_accessor_param().delta_keep_days();
    auto delete_threshold = _config.downpour_accessor_param().delete_threshold();

    // time_decay first
    DownpourFeatureValue::show(value) *= _show_click_decay_rate;
    DownpourFeatureValue::click(value) *= _show_click_decay_rate;

    // shrink after
    auto score = show_click_score(DownpourFeatureValue::show(value), DownpourFeatureValue::click(value));
    auto unseen_days = DownpourFeatureValue::unseen_days(value);
    if (score < delete_threshold || unseen_days > delta_keep_days) {
        return true;
    }
    return false;
}

bool DownpourFeatureValueAccessor::save(float* value, int param) {
    //auto base_threshold = _config.downpour_accessor_param().base_threshold();
    //auto delta_threshold = _config.downpour_accessor_param().delta_threshold();
    //auto delta_keep_days = _config.downpour_accessor_param().delta_keep_days();
    auto base_threshold = _config.downpour_accessor_param().base_threshold();
    auto delta_threshold = _config.downpour_accessor_param().delta_threshold();
    auto delta_keep_days = _config.downpour_accessor_param().delta_keep_days();
    if (param == 2) {
        delta_threshold = 0;
    }
    switch (param) {
        // save all
        case 0:
            {
                return true;
            }
        // save xbox delta
        case 1:
        // save xbox base
        case 2:
            {
                if (show_click_score(DownpourFeatureValue::show(value), DownpourFeatureValue::click(value)) >= base_threshold
                        && DownpourFeatureValue::delta_score(value) >= delta_threshold
                        && DownpourFeatureValue::unseen_days(value) <= delta_keep_days) {
                    DownpourFeatureValue::delta_score(value) = 0;
                    return true;
                } else {
                    return false;
                }
            }
        // already decayed in shrink 
        case 3:
            {
                //DownpourFeatureValue::show(value) *= _show_click_decay_rate;
                //DownpourFeatureValue::click(value) *= _show_click_decay_rate;
                DownpourFeatureValue::unseen_days(value)++;
                return true;
            }
        default:
            return true;
    };
}

int32_t DownpourFeatureValueAccessor::create(float** values, size_t num) {
    auto embedx_dim = _config.embedx_dim();
    for (size_t value_item = 0; value_item < num; ++value_item) {
        float* value = values[value_item];
        value[DownpourFeatureValue::unseen_days_index()] = 0;
        value[DownpourFeatureValue::delta_score_index()] = 0;
        value[DownpourFeatureValue::show_index()] = 0;
        value[DownpourFeatureValue::click_index()] = 0;
        _embed_sgd_rule.init_value(1, 
            value + DownpourFeatureValue::embed_w_index(), 
            value[DownpourFeatureValue::embed_g2sum_index()], true);
        _embedx_sgd_rule.init_value(embedx_dim, 
            value + DownpourFeatureValue::embedx_w_index(), 
            value[DownpourFeatureValue::embedx_g2sum_index()]);
    }
    return 0;
}

bool DownpourFeatureValueAccessor::need_extend_mf(float* value) {
    float show = value[DownpourFeatureValue::show_index()];
    float click = value[DownpourFeatureValue::click_index()];
    //float score = (show - click) * _config.downpour_accessor_param().nonclk_coeff()
    float score = (show - click) * _config.downpour_accessor_param().nonclk_coeff()
        + click * _config.downpour_accessor_param().click_coeff();
        //+ click * _config.downpour_accessor_param().click_coeff();
    return score >= _config.embedx_threshold();
}

// from DownpourFeatureValue to DownpourPullValue
int32_t DownpourFeatureValueAccessor::select(float** select_values, const float** values, size_t num) {
    auto embedx_dim = _config.embedx_dim();
    for (size_t value_item = 0; value_item < num; ++value_item) {
        float* select_value = select_values[value_item];
        float* value = const_cast<float*>(values[value_item]);
        select_value[DownpourPullValue::show_index()] = value[DownpourFeatureValue::show_index()];
        select_value[DownpourPullValue::click_index()] = value[DownpourFeatureValue::click_index()];
        select_value[DownpourPullValue::embed_w_index()] = value[DownpourFeatureValue::embed_w_index()];
        memcpy(select_value + DownpourPullValue::embedx_w_index(), 
            value + DownpourFeatureValue::embedx_w_index(), embedx_dim * sizeof(float));
    }
    return 0;
}

// from DownpourPushValue to DownpourPushValue
// first dim: item
// second dim: field num
int32_t DownpourFeatureValueAccessor::merge(float** update_values, const float** other_update_values, size_t num) {
    auto embedx_dim = _config.embedx_dim();
    size_t total_dim = DownpourPushValue::dim(embedx_dim);
    for (size_t value_item = 0; value_item < num; ++value_item) {
        float* update_value = update_values[value_item];
        const float* other_update_value = other_update_values[value_item];
        for (auto i = 0u; i < total_dim; ++i) {
            update_value[i] += other_update_value[i];
        }
    }
    return 0;
}

// from DownpourPushValue to DownpourFeatureValue
// first dim: item
// second dim: field num
int32_t DownpourFeatureValueAccessor::update(float** update_values, const float** push_values, size_t num) {
    auto embedx_dim = _config.embedx_dim();
    for (size_t value_item = 0; value_item < num; ++value_item) {
        float* update_value = update_values[value_item];
        const float* push_value = push_values[value_item];
        float push_show = push_value[DownpourPushValue::show_index()];
        float push_click = push_value[DownpourPushValue::click_index()];
        update_value[DownpourFeatureValue::show_index()] += push_show;
        update_value[DownpourFeatureValue::click_index()] += push_click;
        update_value[DownpourFeatureValue::delta_score_index()] +=
            (push_show - push_click) * _config.downpour_accessor_param().nonclk_coeff() +
            push_click * _config.downpour_accessor_param().click_coeff();
            //(push_show - push_click) * _config.downpour_accessor_param().nonclk_coeff() +
            //push_click * _config.downpour_accessor_param().click_coeff();
        update_value[DownpourFeatureValue::unseen_days_index()] = 0;
        _embed_sgd_rule.update_value(1,
            update_value + DownpourFeatureValue::embed_w_index(),
            update_value[DownpourFeatureValue::embed_g2sum_index()],
            push_value + DownpourPushValue::embed_g_index(), push_show);
        _embedx_sgd_rule.update_value(embedx_dim,
            update_value + DownpourFeatureValue::embedx_w_index(),
            update_value[DownpourFeatureValue::embedx_g2sum_index()],
            push_value + DownpourPushValue::embedx_g_index(), push_show);
    }
    return 0;
}

bool DownpourFeatureValueAccessor::create_value(int stage, const float* value) {
    // stage == 0, pull
    // stage == 1, push
    if (stage == 0) {
        return true;
    } else if (stage == 1) {
        auto show = DownpourPushValue::show(const_cast<float*>(value));
        auto click = DownpourPushValue::click(const_cast<float*>(value));
        auto score = show_click_score(show, click);
        if (score <= 0) {
            return false;
        }
        if (score >= 1) {
            return true;
        }
        return local_uniform_real_distribution<float>()(local_random_engine()) < score;
    } else {
        return true;
    }
}

float DownpourFeatureValueAccessor::show_click_score(float show, float click) {
    //auto nonclk_coeff = _config.downpour_accessor_param().nonclk_coeff();
    //auto click_coeff = _config.downpour_accessor_param().click_coeff();
    auto nonclk_coeff = _config.downpour_accessor_param().nonclk_coeff();
    auto click_coeff = _config.downpour_accessor_param().click_coeff();
    return (show - click) * nonclk_coeff + click * click_coeff;
}

std::string DownpourFeatureValueAccessor::parse_to_string(const float* v, int param) {
    std::ostringstream os;
    os << v[0] << " "
        << v[1] << " "
        << v[2] << " "
        << v[3] << " "
        << v[4] << " "
        << v[5];
    auto show = DownpourFeatureValue::show(const_cast<float*>(v));
    auto click = DownpourFeatureValue::click(const_cast<float*>(v));
    auto score = show_click_score(show, click);
    if (score >= _config.embedx_threshold()) {
        os << " " << v[6];
        for (auto i = 0; i < _config.embedx_dim(); ++i) {
            os << " " << v[7 + i];
        }
    }
    return os.str();
}

int DownpourFeatureValueAccessor::parse_from_string(const std::string& str, float* value) {
    _embedx_sgd_rule.init_value(_config.embedx_dim(), 
        value + DownpourFeatureValue::embedx_w_index(), 
        value[DownpourFeatureValue::embedx_g2sum_index()]);
    auto ret = str_to_float(str.data(), value);
    CHECK(ret >= 6) << "expect more than 6 real:" << ret;
    return 0;
}

size_t DownpourSparseValueAccessor::dim() {
    return _config.embedx_dim();
}

size_t DownpourSparseValueAccessor::size() {
    return dim() * sizeof(float);
}

// pull value
size_t DownpourSparseValueAccessor::select_dim() {
    return dim();
}

size_t DownpourSparseValueAccessor::select_size() {
    return size();
}

// push value
size_t DownpourSparseValueAccessor::update_dim() {
    return dim();
}

size_t DownpourSparseValueAccessor::update_size() {
    return size();
}

int32_t DownpourSparseValueAccessor::create(float** value, size_t num) {
    auto embedx_dim = _config.embedx_dim();
    _embedx_sgd_rule.init_value(embedx_dim, num, value);
    return 0;
}

int32_t DownpourSparseValueAccessor::select(float** select_values, const float** values, size_t num) {
    auto embedx_dim = _config.embedx_dim();
    for (auto i = 0u; i < embedx_dim; ++i) {
        memcpy(select_values[i], values[i], sizeof(float) * num);
    }
    return 0;
}

int32_t DownpourSparseValueAccessor::merge(float** update_values, const float** other_update_values, size_t num) {
    for (auto i = 0u; i < dim(); ++i) {
        Eigen::Map<Eigen::MatrixXf> mat(update_values[i], 1, num);
        Eigen::Map<const Eigen::MatrixXf> mat2(other_update_values[i], 1, num);

        mat += mat2;
    }
    return 0;
}

int32_t DownpourSparseValueAccessor::update(float** update_values, const float** push_values, size_t num) {
    auto embedx_dim = _config.embedx_dim();
    _embedx_sgd_rule.update_value(embedx_dim, num, update_values, push_values);
    return 0;
}

std::string DownpourSparseValueAccessor::parse_to_string(const float* value, int param) {
    thread_local std::ostringstream os;
    os.str("");
    for (auto i = 0u; i < dim(); ++i) {
        os << value[i];
        if (i != dim()) {
            os << " ";
        }
    }
    return os.str();
}

int DownpourSparseValueAccessor::parse_from_string(const std::string& str, float* value) {
    auto index = str_to_float(str.data(), value);
    CHECK(index == dim()) << "reading feature size not match expect[" 
        << dim() << "] real[" << index << "]";
    return 0;
}

int DownpourDenseValueAccessor::initialize() {
    auto name = _config.dense_sgd_param().name();
    _sgd_rule = global_dense_sgd_rule_factory().produce(name);
    _sgd_rule->load_config(_config.dense_sgd_param());

    return 0;
}

size_t DownpourDenseValueAccessor::dim() {
    return _sgd_rule->dim();
}

size_t DownpourDenseValueAccessor::dim_size(size_t dim) {
    return _sgd_rule->dim_size(dim);
}

size_t DownpourDenseValueAccessor::size() {
    return _sgd_rule->size();
}

int32_t DownpourDenseValueAccessor::create(float** value, size_t num) {
    _sgd_rule->init_value(value, num);
    return 0;
}

// pull from SGDValue to float
int32_t DownpourDenseValueAccessor::select(
    float** select_values, const float** values, size_t num) {
    _sgd_rule->get_weight(select_values, values, num);
    return 0;
}

// merge from gradient to gradient
int32_t DownpourDenseValueAccessor::merge(
    float** update_values, const float** other_update_values, size_t num) {
    //_sgd_rule->merge_value(update_values, other_update_values, num);
    Eigen::Map<Eigen::MatrixXf> u_mat(update_values[0], 1, num);
    Eigen::Map<const Eigen::MatrixXf> o_mat(other_update_values[0], 1, num);
    u_mat += o_mat;
    return 0;
}

// push from float to SGDValue
int32_t DownpourDenseValueAccessor::update(
    float** values, const float** update_values, size_t num){
    _sgd_rule->update_weight(values, update_values, num);

    return 0;
}

int DownpourDenseValueAccessor::set_weight(float** values, const float** update_values, size_t num) {
    _sgd_rule->set_weight(update_values, values, num);
    return 0;
}

std::string DownpourDenseValueAccessor::parse_to_string(const float* v, int param) {
    return _sgd_rule->to_string(v);
}
int DownpourDenseValueAccessor::parse_from_string(const std::string& str, float* v) {
    return _sgd_rule->from_string(str, v);
}
}
}
