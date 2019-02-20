#pragma once

#include <sys/types.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <fcntl.h>
#include <string>
#include <memory>
#include <sys/syscall.h>
#include "glog/logging.h"
#include "ps_string.h"

namespace paddle {
namespace ps {

inline bool& shell_verbose_internal() {
    static bool x = false;
    return x;
}

inline bool shell_verbose() {
    return shell_verbose_internal();
}

inline void shell_set_verbose(bool x) {
    shell_verbose_internal() = x;
}

extern std::shared_ptr<FILE> shell_fopen(const std::string& path, const std::string& mode);

extern std::shared_ptr<FILE> shell_popen(const std::string& cmd, const std::string& mode, int* err_no);

extern std::pair<std::shared_ptr<FILE>, std::shared_ptr<FILE>> shell_p2open(const std::string& cmd);

inline void shell_execute(const std::string& cmd) {
    int err_no = 0;
    do {
    	err_no = 0;
        shell_popen(cmd, "w", &err_no);
    } while (err_no == -1);
}

extern std::string shell_get_command_output(const std::string& cmd);

} //namespace ps
} //namespace paddle
