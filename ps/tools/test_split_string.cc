#include "ps_instance.h"
#include "communicate/ps_client.h"
#include "communicate/ps_server.h"
#include "common/afs_warpper.h"
#include "common/fs.h"
#include "common/ps_string.h"
#include "proto/ps.pb.h"

int main(int argc, char *argv[])
{
    // test split_string
    std::string test_str = "x y z o p q";
    std::vector<std::string> res_list = paddle::ps::split_string(test_str, " ");
    for (int i = 0; i  < res_list.size(); ++i) {
        std::cerr << res_list[i] << std::endl;
    }

    // test split_string without space on boundary
    std::string test_str2 = "  test split_string without space on boundary    ";
    std::vector<std::string> res_list2 = paddle::ps::split_string(test_str2);
    for (int i = 0; i < res_list2.size(); ++i) {
        std::cerr << res_list2[i] << std::endl;
    }

    return 0;
}
