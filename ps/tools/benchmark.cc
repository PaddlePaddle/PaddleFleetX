#include <fcntl.h>
#include <fstream>
#include <sstream>  
#include "proto/ps.pb.h"
#include "common/thread_pool.h"
#include "communicate/ps_env.h"
#include "communicate/ps_client.h"
#include "communicate/ps_server.h"
#include "json2pb/json_to_pb.h"
#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

DEFINE_string(pslib_conf_path, "./example/server_test.proto", "conf for server_test");
DEFINE_int32(pslib_run_mode, 0, "0:client 1:server");
DEFINE_int32(pslib_server_num, 50, "start num of servers");
DEFINE_int32(pslib_client_num, 1, "start num of clients");
DEFINE_string(pslib_server_list, "", "signs of server_list, split by ','");
DEFINE_int32(pslib_client_request_rounds, 10, "request rounds for client");
DEFINE_int32(pslib_client_request_qps, 1, "qps per request rounds");
DEFINE_int32(pslib_client_thread_pool_size, 10, "thread size for client to send request");
DEFINE_int32(pslib_client_request_type, 0, "0:push_dense 1:pull_dense 2:push_sparse 3:pull_sparse 4:pull_all 5:client2client");
DEFINE_int32(pslib_client_sparse_num_per_request, 70000, "sparse value num per request");

using namespace paddle;
using namespace paddle::ps;
int start_client(paddle::PSParameter&);
int start_server(paddle::PSParameter&);

int main(int argc, char *argv[]) {
    ::google::ParseCommandLineFlags(&argc, &argv, true);
    google::SetCommandLineOption("flagfile", "conf/gflags.conf");
    paddle::PSParameter ps_parameter;
    int file_descriptor = open(FLAGS_pslib_conf_path.c_str(), O_RDONLY);
    if (file_descriptor == -1){
        LOG(ERROR) << "FATAL: cant open " << FLAGS_pslib_conf_path;
    }
    google::protobuf::io::FileInputStream fileInput(file_descriptor);
    if (!google::protobuf::TextFormat::Parse(&fileInput, &ps_parameter)) {
        LOG(ERROR) << "FATAL: fail to parse " << FLAGS_pslib_conf_path;
        exit(-1);
    }
    close(file_descriptor); 
    if (FLAGS_pslib_run_mode == 0) {
        start_client(ps_parameter);
    }
    if (FLAGS_pslib_run_mode == 1) {
        start_server(ps_parameter);
    }
    return 0;
}

enum RequestType {
    PslibRequestPushDense = 0,
    PslibRequestPullDense = 1,
    PslibRequestPushSparse = 2,
    PslibRequestPullSparse = 3,
    PslibRequestPushAll = 4,
    PslibClient2ClientMsg = 5
};

Region* make_dense_region(int table_id, size_t size) {
    static std::mutex mutex;
    static Region table_regions[100];
    if (table_regions[table_id].data != NULL) {
        return &(table_regions[table_id]);
    }
    mutex.lock();
    if (table_regions[table_id].data == NULL) {
        float* data = new float[size];
        for (size_t i = 0; i < size; ++i) {
            data[i] = i * 0.001;
        }
        table_regions[table_id].data = (char*)data; 
        table_regions[table_id].size = size * sizeof(float);
    }
    mutex.unlock();
    return &(table_regions[table_id]);
}

uint64_t* make_sparse_keys() {
    static std::mutex mutex;
    static uint64_t* sparse_keys = NULL;
    if (sparse_keys != NULL) {
        return sparse_keys;
    }
    mutex.lock();
    if (sparse_keys == NULL) {
        auto hasher = std::hash<int>();
        sparse_keys = new uint64_t[FLAGS_pslib_client_sparse_num_per_request];
        for (int i = 0; i < FLAGS_pslib_client_sparse_num_per_request; ++i) {
            sparse_keys[i] = i * 100 + 1 + hasher(i);
        }
    }
    mutex.unlock();
    return sparse_keys;
}

float** make_sparse_values(int table_id, size_t data_size) {
    static std::mutex mutex;
    static bool table_is_init[100] = {false};
    static float** table_values[100] = {NULL};
    if (table_is_init[table_id]) {
        return table_values[table_id];
    }
    mutex.lock();
    if (table_is_init[table_id] == false) {
        table_values[table_id] = new float*[FLAGS_pslib_client_sparse_num_per_request];
        float** values = table_values[table_id];
        for (int i = 0; i < FLAGS_pslib_client_sparse_num_per_request; ++i) {
            values[i] = new float[data_size];
            for (size_t j = 0; j < data_size; ++j) {
                values[i][j] = i * 0.001 + (j * i + j) * 0.2323;
            }
        }
        table_is_init[table_id] = true;
    }
    mutex.unlock();
    return table_values[table_id];
}

//void async_request_server(PSClient* client, const DownpourServerParameter* server_param) {
void async_request_server(PSClient* client, const DownpourServerParameter* server_param) {
    if (FLAGS_pslib_client_request_type == PslibClient2ClientMsg) {
        static std::string msg("HelloWorld");
        client->send_client2client_msg(0, 0, msg);
        return;
    }
    //size_t table_size = server_param->downpour_table_param_size();
    size_t table_size = server_param->downpour_table_param_size();
    for (int i = 0; i < table_size; ++i) {
        //const auto& table_param = server_param->downpour_table_param(i);
        const auto& table_param = server_param->downpour_table_param(i);
        int table_id = table_param.table_id();
        size_t fea_size = table_param.accessor().fea_dim();
        if (table_param.type() == PS_DENSE_TABLE) {
            auto* region = make_dense_region(table_id, fea_size);
            if (FLAGS_pslib_client_request_type == PslibRequestPushAll ||
                FLAGS_pslib_client_request_type == PslibRequestPushDense) {
                client->push_dense(region, 1, table_id);
            } else if (FLAGS_pslib_client_request_type == PslibRequestPullDense) {
                client->pull_dense(region, 1, table_id);
            }
        } else if (table_param.type() == PS_SPARSE_TABLE) {
            if (FLAGS_pslib_client_request_type == PslibRequestPushAll ||
                FLAGS_pslib_client_request_type == PslibRequestPushSparse) {
                client->push_sparse(table_id, 
                    make_sparse_keys(), 
                    (const float**) make_sparse_values(table_id, fea_size), 
                    FLAGS_pslib_client_sparse_num_per_request); 
            } else if (FLAGS_pslib_client_request_type == PslibRequestPullSparse) {
                client->pull_sparse(make_sparse_values(table_id, fea_size), 
                    table_id, 
                    make_sparse_keys(), 
                    FLAGS_pslib_client_sparse_num_per_request); 
            }
        }
    }
}

int start_client(paddle::PSParameter& param) {
    if (FLAGS_pslib_server_list.empty()) {
        LOG(ERROR) << "FLAGS_pslib_server_list should not be empty";
        return -1;
    }
    char* str_end = NULL;
    LocalPSEnvironment env;
    std::vector<std::string> server_signs = split_string(FLAGS_pslib_server_list, ",");
    int rank_idx = 0;
    for (const auto& server_sign : server_signs) {
        PSHost host;
        if (server_sign.find(":") != std::string::npos) {
            host.rank = rank_idx;
            host.ip = server_sign.substr(0, server_sign.find(":"));
            host.port = atoi(server_sign.c_str() + server_sign.find(":") + 1);
        } else {
            host.parse_from_uint64(strtoull(server_sign.c_str(), &str_end, 10));
        }
        env.registe_ps_server(host.ip, host.port, host.rank);
        ++rank_idx;
    }
    env.gather_ps_servers();
    
    std::shared_ptr<WorkerPool> thread_pool = 
        std::make_shared<WorkerPool>(FLAGS_pslib_client_thread_pool_size);    
    
    std::vector<std::shared_ptr<PSClient>> client_list;
    client_list.resize(FLAGS_pslib_client_num);
    for (int i = 0; i < FLAGS_pslib_client_num; ++i) {
        client_list[i].reset(PSClientFactory::create(param));
        client_list[i]->configure(param, env, i);
        client_list[i]->registe_client2client_msg_handler(
            0, [](int msg_type, int client_id, const std::string& msg) -> int32_t {
            LOG(WARNING) << "receive type:" << msg_type << ", from_id:" << client_id << ", msg:" << msg;
            return 0;
        });
    }

    //const auto& server_param = param.server_param().downpour_server_param();
    const auto& server_param = param.server_param().downpour_server_param();
    
    for (int i = 0; i < FLAGS_pslib_client_request_rounds; ++i) {
        for (int j = 0; j < FLAGS_pslib_client_request_qps; ++j) {
            thread_pool->AddTask(async_request_server, 
                client_list[j % FLAGS_pslib_client_num].get(), &server_param);
            usleep(1000000 / FLAGS_pslib_client_request_qps);        
        }
        //sleep(1);
    }
    LOG(INFO) << "client flush data";
    for (int i = 0; i < FLAGS_pslib_client_num; ++i) {
        client_list[i]->flush();
    }
    LOG(INFO) << "request all end";
    return 0;
}

int start_server(paddle::PSParameter& param) {
    LocalPSEnvironment env;
    std::vector<std::shared_ptr<PSServer>> server_list;
    server_list.resize(FLAGS_pslib_server_num);
    for (int i = 0; i < FLAGS_pslib_server_num; ++i) {
        server_list[i].reset(PSServerFactory::create(param));
        server_list[i]->configure(param, env, i);
        server_list[i]->start();
    }
    env.gather_ps_servers();
    while (true) {
        LOG(INFO) << "server is running..";
        sleep(10);
    }
    return 0;
}
