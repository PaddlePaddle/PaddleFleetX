#include "common/registerer.h"

namespace paddle {
namespace ps {
BaseClassMap& global_factory_map() {
    static BaseClassMap *base_class = new BaseClassMap();
    return *base_class;
}

BaseClassMap& global_factory_map_cpp() {
    return global_factory_map();
}

}// namespace ps
}// namespace paddle

/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */
