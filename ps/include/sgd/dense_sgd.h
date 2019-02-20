#ifndef BAIDU_BAIDU_PSLIB_INCLUDE_SGD_DENSE_SGD_H
#define BAIDU_BAIDU_PSLIB_INCLUDE_SGD_DENSE_SGD_H
#include <string>
#include <sstream>
#include "proto/ps.pb.h"
#include "common/ps_string.h"

namespace paddle {
namespace ps {

class DenseSGDRule {
public:
    virtual ~DenseSGDRule() {}
    virtual void load_config(const DenseSGDRuleParameter& param) {
        _name = param.name();
    }
    virtual size_t dim() = 0;
    virtual size_t dim_size(size_t dim) = 0;
    virtual size_t size() = 0;
    virtual void get_weight(float** value, const float** from, size_t num) = 0;
    virtual void set_weight(const float** w, float** value, size_t num) = 0;
    virtual float get_avg_weight(float* value) = 0;
    virtual void set_avg_weight(float avg_w, float* value) = 0;
    virtual void update_weight(float** value, const float** grad, size_t num) = 0;
    virtual void init_value(float** value, size_t num) = 0;
    virtual void merge_value(float** value, const float** from, size_t num) = 0;
    virtual const std::string& get_sgd_type() const  = 0;
    virtual const std::string& get_sgd_class() const  = 0;
    virtual std::string to_string(const float* value) = 0;
    virtual int from_string(const std::string& str, float* value) = 0;
    const std::string& get_name() const {
        return _name;
    }
private:
    std::string _name;
};

inline std::vector<float>& adam_local_scale() {
    thread_local std::vector<float> vec;
    return vec;
}

struct AdamSGDValue {
    /*
    float w;
    float avg_w;
    float ada_d2sum;
    float ada_g2sum;
    float mom_velocity;
    */
    static std::string to_string(const float* value) {
        std::ostringstream os;
        os << value[0] << " "
            << value[1] << " "
            << value[2] << " "
            << value[3] << " "
            << value[4];
        return os.str();
    }

    static int dim() {
        return 5;
    }
    static int w_index() {
        return 0;
    }
    static int avg_w_index() {
        return AdamSGDValue::w_index() + 1;
    }
    static int ada_d2sum_index() {
        return AdamSGDValue::avg_w_index() + 1;
    }
    static int ada_g2sum_index() {
        return AdamSGDValue::ada_d2sum_index() + 1;
    }
    static int mom_velocity_index() {
        return AdamSGDValue::ada_g2sum_index() + 1;
    }
    static float& w(float* val) {
        return val[0];
    } 
    static float& avg_w(float* val) {
        return val[1];
    }
    static float& ada_d2sum(float* val) {
       return val[2];
    }
    static float& ada_g2sum(float* val) {
        return val[3];
    }
    static float& mom_velocity(float* val) {
        return val[4];
    }
};

class AdamSGDRule : public DenseSGDRule {
public:
    size_t dim() {return 5;}
    size_t dim_size(size_t dim) {return sizeof(float);}
    size_t size() {return sizeof(float) * 5;}
    void load_config(const DenseSGDRuleParameter& param);
    void init_value(float** value, size_t num);
    void merge_value(float** value, const float** from, size_t num);
    void get_weight(float** value, const float** from, size_t num);
    void set_weight(const float** w, float** value, size_t num);
    float get_avg_weight(float* value);
    void set_avg_weight(float avg_w, float* value);
    void update_weight(float** value, const float** grad, size_t num);
    const std::string& get_sgd_type() const;
    const std::string& get_sgd_class() const;
    std::string to_string(const float* v);
    int from_string(const std::string& str, float* v);

public:
    double learning_rate;
    double avg_decay_rate;
    double ada_decay_rate;
    double ada_epsilon;
    double mom_decay_rate;
};

struct NaiveSGDValue {
    /*
    float w;
    */

    static std::string to_string(const float* value) {
        return std::to_string(value[0]);
    }
    static float& w(float* v) {
        return v[0];
    }
};

class NaiveSGDRule : public DenseSGDRule {
public:
    size_t dim() {return 1;}
    size_t dim_size(size_t dim) {return sizeof(float);}
    size_t size() {return sizeof(float);}
    void load_config(const DenseSGDRuleParameter& param);
    void init_value(float** value, size_t num);
    void merge_value(float** value, const float** from, size_t num);
    void get_weight(float** value, const float** from, size_t num);
    void set_weight(const float** w, float** value, size_t num);
    float get_avg_weight(float* value);
    void set_avg_weight(float avg_w, float* value);
    void update_weight(float** value, const float** grad, size_t num);
    const std::string& get_sgd_type() const;
    const std::string& get_sgd_class() const;

    std::string to_string(const float* v);
    int from_string(const std::string& str, float* v);
public:
    double learning_rate;
    double avg_decay_rate;
};

struct SummarySGDValue {
    //float w;
    static float& w(float* v) {
        return v[0];
    }
    static std::string to_string(const float* v) {
        return std::to_string(v[0]);
    }
};

class SummarySGDRule : public DenseSGDRule {
public:
    size_t dim() {return 1;}
    size_t dim_size(size_t dim) {return sizeof(float);}
    size_t size() {return sizeof(float) * 1;}
    void load_config(const DenseSGDRuleParameter& param);
    void init_value(float** value, size_t num);
    void merge_value(float** value, const float** from, size_t num);
    void get_weight(float** value, const float** from, size_t num);
    void set_weight(const float** w, float** value, size_t num);
    float get_avg_weight(float* value);
    void set_avg_weight(float avg_w, float* value);
    void update_weight(float** value, const float** grad, size_t num);
    const std::string& get_sgd_type() const;
    const std::string& get_sgd_class() const;

    std::string to_string(const float* v);
    int from_string(const std::string& str, float* v);
public:
    double decay_rate;
};

class MovingAverageRule: public DenseSGDRule {
public:
    size_t dim() {return 1;}
    size_t dim_size(size_t dim) {return sizeof(float);}
    size_t size() {return sizeof(float) * 1;}

    void load_config(const DenseSGDRuleParameter& param);
    void init_value(float** value, size_t num);
    void merge_value(float** value, const float** from, size_t num);
    void get_weight(float** value, const float** from, size_t num);
    void set_weight(const float** w, float** value, size_t num);
    float get_avg_weight(float* value);
    void set_avg_weight(float avg_w, float* value);
    void update_weight(float** value, const float** grad, size_t num);
    const std::string& get_sgd_type() const;
    const std::string& get_sgd_class() const;

    std::string to_string(const float* v);
    int from_string(const std::string& str, float* v);
public:
    double decay_rate;
};

}
}
#endif
