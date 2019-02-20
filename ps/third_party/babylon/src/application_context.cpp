#include "application_context.h"

namespace baidu {
namespace feed {
namespace mlarch {
namespace babylon {

const ::std::string& ApplicationContext::ComponentHolder::type() const noexcept {
    static ::std::string dummy_type("<unknown>");
    return dummy_type;
}

int32_t ApplicationContext::ComponentHolder::initialize() noexcept {
    return 0;
}

int32_t ApplicationContext::ComponentHolder::wireup(ApplicationContext&) noexcept {
    return 0;
}

}  // babylon
}  // mlarch
}  // feed
}  // baidu
