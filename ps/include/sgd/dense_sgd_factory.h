#ifndef BAIDU_BAIDU_PSLIB_INCLUDE_SGD_DENSE_SGD_FACTORY_H
#define BAIDU_BAIDU_PSLIB_INCLUDE_SGD_DENSE_SGD_FACTORY_H
#include "common/factory.h"
#include "dense_sgd.h"
namespace paddle {
namespace ps {
    
inline Factory<DenseSGDRule>& global_dense_sgd_rule_factory() {
    static Factory<DenseSGDRule> f;
    return f;
}

inline void pslib_sgd_init() {
    static bool sgd_initial = false;

    if (sgd_initial) {
        return;
    }

    sgd_initial = true;
    
    Factory<DenseSGDRule>& factory = global_dense_sgd_rule_factory();
    factory.add<AdamSGDRule>("adam");
    factory.add<NaiveSGDRule>("naive");
    factory.add<SummarySGDRule>("summary");
    factory.add<MovingAverageRule>("moving_average");
}

}
}
#endif
