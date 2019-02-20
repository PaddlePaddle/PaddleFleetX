#pragma once

#include <functional>
#include <time.h>
#include "glog/logging.h"
#include "ps_string.h"
#include "shell.h"

namespace paddle {
namespace ps {

// Call func(args...). If interrupted by signal, recall the function.
template<class FUNC, class... ARGS>
auto ignore_signal_call(FUNC && func, ARGS
        && ... args) -> typename std::result_of<FUNC(ARGS...)>::type {
    for (;;) {
        auto err = func(args...);

        if (err < 0 && errno == EINTR) {
            LOG(INFO) << "Signal is caught. Ignored.";
            continue;
        }

        return err;
    }
}

// note: this implementation must be replaced with std::make_unique when c++14 is enabled.
template<typename T, typename... Args>
std::unique_ptr<T> make_unique(Args&&... args) {
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

//20160912 -> 20160913
inline std::string get_next_date_string(const std::string& curr_date) {
    CHECK(!curr_date.empty()) << "curr_date is empty";
    std::string cmd = format_string("date -d '%s 1 day' +%%Y%%m%%d", curr_date.c_str());
    return trim_spaces(shell_get_command_output(cmd));
}

}
}
