#pragma once
#include <vector>
#include <stdint.h>
#include "proto/ps.pb.h"
#include "accessor.h"
#include "common/registerer.h"

namespace paddle {
namespace ps {

class SimpleCopyValueAccessor : public ValueAccessor {
public:
    SimpleCopyValueAccessor() {}
    virtual ~SimpleCopyValueAccessor() {}
    virtual int initialize() { return 0; }
    virtual size_t dim() { return 15; }
    virtual size_t dim_size(size_t dim) { return sizeof(float); }
    virtual size_t size() { return dim() * sizeof(float); }
    virtual size_t select_dim() {return dim();}
    virtual size_t select_dim_size(size_t dim) {return dim_size(dim);}
    virtual size_t select_size() {return size();}
    virtual size_t update_dim() {return dim();}
    virtual size_t update_dim_size(size_t dim) {return dim_size(dim);}
    virtual size_t update_size() {return size();}
    virtual bool shrink(float* /*value*/) {
        return false;
    } 
    virtual bool save(float* /*value*/, int /*param*/) {
        return true;
    }
    virtual int32_t create(float** value, size_t num) {
        for (int i = 0; i < size() / sizeof(float); ++i) {
            memset(value[i], 0, num * sizeof(float));
        }
        return 0;
    }
    virtual int32_t select(float** select_values, const float** values, size_t num) {
        for (int i = 0; i < select_size() / sizeof(float); ++i) {
            memcpy(select_values[i], values[i], num * sizeof(float));
        }
        return 0;
    }
    virtual int32_t merge(float** update_values, const float** other_update_values, size_t num) {
        for (int i = 0; i < update_size() / sizeof(float); ++i) {
            for (int j = 0; j < num; ++j) {
                update_values[i][j] += other_update_values[i][j];
            }
        }
        return 0;
    }
    virtual int32_t update(float** values, const float** update_values, size_t num) {
        for (int i = 0; i < size()/sizeof(float); ++i) {
            for (int j = 0; j < num; ++j) {
                values[i][j] += update_values[i][j];
            }
        }
        return 0;
    }
    virtual std::string parse_to_string(const float* value, int param) {
        return "";
    }
    virtual int32_t parse_from_string(const std::string& data, float* value) {
        return 0;
    }
};

}
}
