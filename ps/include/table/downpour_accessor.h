#pragma once
#include <vector>
#include <stdio.h>
#include <stdint.h>
#include "proto/ps.pb.h"
#include "accessor.h"
#include "sgd/sparse_sgd.h"
#include "common/registerer.h"

namespace paddle {
namespace ps {
class DenseSGDRule;

class DownpourFeatureValueAccessor : public ValueAccessor {
public:
    struct DownpourFeatureValue {
        /*
        float unseen_days;
        float delta_score;
        float show;
        float click;
        float embed_w;
        float embed_g2sum;
        float embedx_g2sum;
        std::vector<float> embedx_w; 
        */

        static int dim(int embedx_dim) {
            return 7 + embedx_dim;
        }
        static int dim_size(size_t dim, int embedx_dim) {
            return sizeof(float);
        }
        static int size(int embedx_dim) {
            return dim(embedx_dim) * sizeof(float);
        }
        static int unseen_days_index() {
            return 0;
        }
        static int delta_score_index() {
            return DownpourFeatureValue::unseen_days_index() + 1;
        }
        static int show_index() {
            return DownpourFeatureValue::delta_score_index() + 1;
        } 
        static int click_index() {
            return DownpourFeatureValue::show_index() + 1;
        }
        static int embed_w_index() {
            return DownpourFeatureValue::click_index() + 1;
        }
        static int embed_g2sum_index() {
            return DownpourFeatureValue::embed_w_index() + 1;
        }
        static int embedx_g2sum_index() {
            return DownpourFeatureValue::embed_g2sum_index() + 1;
        }
        static int embedx_w_index() {
            return DownpourFeatureValue::embedx_g2sum_index() + 1;
        }
        static float& unseen_days(float* val) {
            return val[DownpourFeatureValue::unseen_days_index()];
        }
        static float& delta_score(float* val) {
            return val[DownpourFeatureValue::delta_score_index()];
        }
        static float& show(float* val) {
            return val[DownpourFeatureValue::show_index()];
        }
        static float& click(float* val) {
            return val[DownpourFeatureValue::click_index()];
        }
        static float& embed_w(float* val) {
            return val[DownpourFeatureValue::embed_w_index()];
        }
        static float& embed_g2sum(float* val) {
            return val[DownpourFeatureValue::embed_g2sum_index()];
        }
        static float& embedx_g2sum(float* val) {
            return val[DownpourFeatureValue::embedx_g2sum_index()];
        }
        static float* embedx_w(float* val) {
            return (val + DownpourFeatureValue::embedx_w_index());
        }
    };

    struct DownpourPushValue {
        /*
        float show;
        float click;
        float embed_g;
        std::vector<float> embedx_g;
        */

        static int dim(int embedx_dim) {
            return 3 + embedx_dim;
        }

        static int dim_size(int dim, int embedx_dim) {
            return sizeof(float);
        }
        static int size(int embedx_dim) {
            return dim(embedx_dim) * sizeof(float);
        }
        static int show_index() {
            return 0;
        }
        static int click_index() {
            return DownpourPushValue::show_index() + 1;
        }
        static int embed_g_index() {
            return DownpourPushValue::click_index() + 1;
        }
        static int embedx_g_index() {
            return DownpourPushValue::embed_g_index() + 1;
        }
        static float& show(float* val) {
            return val[0];
        }
        static float& click(float* val) {
            return val[1];
        }
        static float& embed_g(float* val) {
            return val[2];
        }
        static float* embedx_g(float* val) {
            return val + 3;
        }
    };

    struct DownpourPullValue {
        /*
        float show;
        float click;
        float embed_w;
        std::vector<float> embedx_w;
        */

        static int dim(int embedx_dim) {
            return 3 + embedx_dim;
        }
        static int dim_size(size_t dim) {
            return sizeof(float);
        }
        static int size(int embedx_dim) {
            return dim(embedx_dim) * sizeof(float); 
        }
        static int show_index() {
            return 0;
        }
        static int click_index() {
            return 1;
        }
        static int embed_w_index() {
            return 2;
        }
        static int embedx_w_index() {
            return 3;
        }
        static float& show(float* val) {
            return val[DownpourPullValue::show_index()];
        }
        static float& click(float* val) {
            return val[DownpourPullValue::click_index()];
        }
        static float& embed_w(float* val) {
            return val[DownpourPullValue::embed_w_index()];
        }
        static float* embedx_w(float* val) {
            return val + DownpourPullValue::embedx_w_index();
        }
    };
    DownpourFeatureValueAccessor() {}
    virtual ~DownpourFeatureValueAccessor() {}
    
    virtual int initialize() {
        _embed_sgd_rule.load_config(_config.sparse_sgd_param());
        _embed_sgd_rule.initial_range = 0.0f;
        _embedx_sgd_rule.load_config(_config.sparse_sgd_param());
        //_show_click_decay_rate = _config.downpour_accessor_param().show_click_decay_rate();
        _show_click_decay_rate = _config.downpour_accessor_param().show_click_decay_rate();
        return 0;
    }
    // value维度
    virtual size_t dim();
    // value各个维度的size
    virtual size_t dim_size(size_t dim);
    // value各维度相加总size
    virtual size_t size();
    // value中mf动态长度部分总size大小, sparse下生效
    virtual size_t mf_size();
    // pull value维度
    virtual size_t select_dim();
    // pull value各个维度的size
    virtual size_t select_dim_size(size_t dim);
    // pull value各维度相加总size
    virtual size_t select_size();
    // push value维度
    virtual size_t update_dim();
    // push value各个维度的size
    virtual size_t update_dim_size(size_t dim);
    // push value各维度相加总size
    virtual size_t update_size();
    // 判断该value是否进行shrink
    virtual bool shrink(float* value); 
    virtual bool need_extend_mf(float* value);
    // 判断该value是否在save阶段dump, param作为参数用于标识save阶段，如downpour的xbox与batch_model
    // param = 0, save all feature
    // param = 1, save delta feature
    // param = 3, save all feature with time decay
    virtual bool save(float* value, int param) override;
    // keys不存在时，为values生成随机值
    // 要求value的内存由外部调用者分配完毕
    virtual int32_t create(float** value, size_t num);
    // 从values中选取到select_values中
    virtual int32_t select(float** select_values, const float** values, size_t num);
    // 将update_values聚合到一起
    virtual int32_t merge(float** update_values, const float** other_update_values, size_t num);
    // 将update_values聚合到一起，通过it.next判定是否进入下一个key
    //virtual int32_t merge(float** update_values, iterator it);
    // 将update_values更新应用到values中
    virtual int32_t update(float** values, const float** update_values, size_t num);

    virtual std::string parse_to_string(const float* value, int param) override;
    virtual int32_t parse_from_string(const std::string& str, float* v) override;
    virtual bool create_value(int type, const float* value);
private:
    float show_click_score(float show, float click);

private:
    SparseSGDRule _embed_sgd_rule;
    SparseSGDRule _embedx_sgd_rule;
    float         _show_click_decay_rate;
};

/** 
 * @brief Accessor for single embedding field
 **/
class DownpourSparseValueAccessor : public ValueAccessor {
/*
     Feature value:
         embed_x
*/
public:
    DownpourSparseValueAccessor() {}
    virtual ~DownpourSparseValueAccessor() {}
    
    virtual int initialize() {
        _embedx_sgd_rule.load_config(_config.sparse_sgd_param());
        return 0;
    }
    // value维度
    virtual size_t dim();
    // value各个维度的size
    virtual size_t dim_size(size_t dim) {
        return sizeof(float);
    }
    // value各维度相加总size
    virtual size_t size();
    // pull value维度
    virtual size_t select_dim();
    // pull value各个维度的size
    virtual size_t select_dim_size(size_t dim) {
        return sizeof(float); 
    }
    // pull value各维度相加总size
    virtual size_t select_size();
    // push value维度
    virtual size_t update_dim();
    // push value各个维度的size
    virtual size_t update_dim_size(size_t dim) {
        return sizeof(float);
    }
    // push value各维度相加总size
    virtual size_t update_size();
    // 判断该value是否进行shrink
    virtual bool shrink(float* value) {
        return false;
    } 
    // 判断该value是否在save阶段dump, param作为参数用于标识save阶段，如downpour的xbox与batch_model
    // param = 0, save all feature
    // param = 1, save delta feature
    // param = 3, save all feature with time decay
    virtual bool save(float* value, int param) {
        return true;
    }
    // keys不存在时，为values生成随机值
    // 要求value的内存由外部调用者分配完毕
    // 批量操作，num: 批量操作的个数
    virtual int32_t create(float** value, size_t num);
    // 从values中选取到select_values中
    virtual int32_t select(float** select_values, const float** values, size_t num);
    // 将update_values聚合到一起
    virtual int32_t merge(float** update_values, const float** other_update_values, size_t num);
    // 将update_values聚合到一起，通过it.next判定是否进入下一个key
    //virtual int32_t merge(float** update_values, iterator it);
    // 将update_values更新应用到values中
    virtual int32_t update(float** values, const float** update_values, size_t num);

    virtual std::string parse_to_string(const float* value, int param);
    virtual int parse_from_string(const std::string& str, float* v);
private:
    SparseSGDRule _embedx_sgd_rule;
};

class DownpourDenseValueAccessor : public ValueAccessor {
public:
    DownpourDenseValueAccessor() {} 
    virtual ~DownpourDenseValueAccessor() {}
    virtual int initialize();
    // value维度
    virtual size_t dim();
    // value各个维度的size
    virtual size_t dim_size(size_t dim);
    // value各维度相加总size
    virtual size_t size();
    // pull value维度
    virtual size_t select_dim() {return 1;}
    // pull value各个维度的size
    virtual size_t select_dim_size(size_t dim) {return sizeof(float);}
    // pull value各维度相加总size
    virtual size_t select_size() {return sizeof(float);}
    // push value维度
    virtual size_t update_dim() {return 1;}
    // push value各个维度的size
    virtual size_t update_dim_size(size_t dim) {return sizeof(float);}
    // push value各维度相加总size
    virtual size_t update_size() {return sizeof(float);}
    // 判断该value是否进行shrink
    virtual bool shrink(float* /*value*/) {
        return false;
    } 
    // 判断该value是否在save阶段dump, param作为参数用于标识save阶段，如downpour的xbox与batch_model
    virtual bool save(float* /*value*/, int /*param*/) {
        return true;
    }
    // keys不存在时，为values生成随机值
    virtual int32_t create(float** value, size_t num);
    // 从values中选取到select_values中
    virtual int32_t select(float** select_values, const float** values, size_t num);
    // 将update_values聚合到一起
    virtual int32_t merge(float** update_values, const float** other_update_values, size_t num);
    // 将update_values聚合到一起，通过it.next判定是否进入下一个key
    //virtual int32_t merge(float** update_values, iterator it);
    // 将update_values更新应用到values中
    virtual int32_t update(float** values, const float** update_values, size_t num);

    virtual int set_weight(float** values, const float** update_values, size_t num);
    virtual std::string parse_to_string(const float* value, int param) override;
    virtual int parse_from_string(const std::string& str, float* v) override;
private:
    std::shared_ptr<DenseSGDRule>   _sgd_rule;
};
}
}
