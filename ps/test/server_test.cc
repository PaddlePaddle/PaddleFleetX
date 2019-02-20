/***************************************************************************
 * 
 * Copyright (c) 2018 Baidu.com, Inc. All Rights Reserved
 * $Id$ 
 * 
 **************************************************************************/
#include <fcntl.h>
#include <fstream>
#include <sstream>  
#include "gtest/gtest.h" 
#include "proto/ps.pb.h"
#include "communicate/downpour_ps_server.h"
#include "communicate/downpour_ps_client.h"
#include "table/downpour_sparse_table.h"
#include "json2pb/json_to_pb.h"
#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
DEFINE_string(pslib_conf_path, "./example/server_test.proto", "conf for server_test");

using namespace paddle;
using namespace paddle::ps;
namespace paddle {
namespace ps {
   
    int accessor_mode = 0; //0:sparse 1:dense
    class DownpourTestAccessor : public ValueAccessor {
        public:
            DownpourTestAccessor() {} 
            virtual ~DownpourTestAccessor() {}
            virtual int initialize() { return 0; }
            virtual size_t dim() { return accessor_mode == 0 ? 3 : 1; }
            virtual size_t dim_size(size_t dim) { return sizeof(float); }
            virtual size_t size() { return dim() * sizeof(float); }
            virtual size_t select_dim() {return accessor_mode == 0 ? 2 : 1;}
            virtual size_t select_dim_size(size_t dim) {return sizeof(float);}
            virtual size_t select_size() {return sizeof(float) * select_dim();}
            virtual size_t update_dim() {return accessor_mode == 0 ? 4 : 1;;}
            virtual size_t update_dim_size(size_t dim) {return sizeof(float);}
            virtual size_t update_size() {return sizeof(float) * update_dim();}
            virtual bool shrink(float* /*value*/) {
                return false;
            } 
            virtual bool save(float* /*value*/, int /*param*/) {
                return true;
            }
            virtual int32_t create(float** value, size_t num) {
                for (int i = 0; i < dim(); ++i) {
                    for (int j = 0; j < num; ++j) {
                        value[i][j] = i * 10 + j;
                    }
                }
                return 0;
            }
            virtual int32_t select(float** select_values, const float** values, size_t num) {
                for (int i = 0; i < select_dim(); ++i) {
                    for (int j = 0; j < num; ++j) {
                        select_values[i][j] = values[i][j];
                    }
                }
                return 0;
            }
            virtual int32_t merge(float** update_values, const float** other_update_values, size_t num) {
                for (int i = 0; i < update_dim(); ++i) {
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
                thread_local std::stringstream ssm;
                ssm.str("");
                for (int i = 0; i < dim(); ++i) {
                    ssm << value[i] << " ";
                }
                return ssm.str();
            }
            
            virtual int32_t parse_from_string(const std::string& data, float* value) {
                char* end_str = NULL;
                value[0] = (float)strtod(data.c_str(), &end_str);
                for (int i = 1; i < dim(); ++i) {
                    value[i] = (float)strtod(end_str, &end_str);
                }
                return 0;
            }
    };
    REGISTER_CLASS(ValueAccessor, DownpourTestAccessor);
}
}

void print_matrix (const char* label, float** data, size_t x, size_t y, uint64_t* keys = NULL) {
    std::cout << label << ":\n";
    for (int i = 0; i < x; ++i) {
        if (keys != NULL) {
            std::cout << keys[i] << ": ";
        }
        for (int j = 0; j < y; ++j) {
            std::cout << data[i][j] << " ";
        }
        std::cout << "\n";
    }
}
TEST(downpour_ps_server, Init) {
    google::SetCommandLineOption("bvar_dump", "true");
    google::SetCommandLineOption("bvar_dump_interval", "1");
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

    LocalPSEnvironment env;
    int server_num = 7;
    for (int i = 0; i < server_num; ++i) {
        auto* server = PSServerFactory::create(ps_parameter);
        ASSERT_TRUE(server != NULL);
        ASSERT_EQ(server->configure(ps_parameter, env, i), 0);
        ASSERT_EQ(server->start(), 0);
        LOG(WARNING) << "Server started in " << server->ip() << ":" << server->port();
    }
    env.gather_ps_servers();
    
    //TODO
    //env.gather_ps_hosts();
    //mpi barrier and gather all server_ip
    
    auto* client = PSClientFactory::create(ps_parameter);
    ASSERT_TRUE(client != NULL);
    ASSERT_EQ(client->configure(ps_parameter, env, 0), 0);

    //test sparse
    uint32_t sparse_kv_num = 1000;
    float** select_values = new float*[sparse_kv_num];
    for (int i = 0; i < sparse_kv_num; ++i) {
        select_values[i] = new float[4];
        for (int j = 0; j < 4; ++j) {
            select_values[i][j] = (i + 1) * j;
        }
    }
    uint64_t* select_keys = new uint64_t[sparse_kv_num];
    for (int i = 0; i < sparse_kv_num; ++i) {
        select_keys[i] = i % 997;
    }
    print_matrix("push_sparse", select_values, 1000, 4, select_keys); 
    for (int i = 0; i < 1000; ++i) {
        client->push_sparse(0, (const uint64_t*)select_keys, (const float**)select_values, 500);
        client->push_sparse(0, (const uint64_t*)select_keys, (const float**)select_values, 997);
        usleep(5000);
    }
    sleep(1);
    
    for (int i = 0; i < sparse_kv_num; ++i) {
        for (int j = 0; j < 3; ++j) {
            select_values[i][j] = 0;
        }
    }
    std::cout << "test test1";
    auto ret = client->pull_sparse(select_values, 0, select_keys, 1000);
    std::cout << "test test2";
    ret.wait();
    std::cout << "test test3";
    print_matrix("pull_sparse", select_values, 1000, 2, select_keys); 
    ASSERT_EQ(select_values[0][0], 0);
    std::cout << "test test";
    //test dense
/*    accessor_mode = 1;
    for (int i = 0; i < sparse_kv_num; ++i) {
        for (int j = 0; j < 4; ++j) {
            select_values[i][j] = (i + 1) * j;
        }
    }
    
    Region* push_regions = new Region[2];
    push_regions[0].data = (char*)select_values[0];
    push_regions[0].size = 3 * sizeof(float);
    push_regions[1].data = (char*)select_values[1];
    push_regions[1].size = 3 * sizeof(float);
    ret = client->push_dense(push_regions, 2, 1);
    ret = client->push_dense(push_regions, 2, 1);
    ret.wait();
    
    Region* pull_regions = new Region[2];
    pull_regions[0].data = (char*)select_values[0];
    pull_regions[0].size = 3 * sizeof(float);
    pull_regions[1].data = (char*)(select_values[1]);
    pull_regions[1].size = 3 * sizeof(float);
    for (int i = 0; i < sparse_kv_num; ++i) {
        for (int j = 0; j < 4; ++j) {
            select_values[i][j] = 0;
        }
    }
    ret = client->pull_dense(pull_regions, 2, 1);
    ret.wait();
    print_matrix("pull_dense", select_values, 2, 3); 
    ASSERT_EQ(select_values[0][0], 0);
    ASSERT_EQ(select_values[0][1], 2);
    ASSERT_EQ(select_values[1][1], 4);
    ASSERT_EQ(select_values[1][2], 8);

    bool test_save_load = false;
    if (test_save_load) {
        ret = client->save("/home/work/rd/xiexionghang/temp/pslib/test/", "");
        ret.wait();

        ret = client->clear();
        ret.wait();

        ret = client->load("/home/work/rd/xiexionghang/temp/pslib/test/", "");
        ret.wait();

        ret = client->pull_dense(pull_regions, 2, 1);
        ret.wait();
        ASSERT_EQ(select_values[0][0], 0);
        ASSERT_EQ(select_values[0][1], 2);
        ASSERT_EQ(select_values[1][1], 4);

        ret = client->pull_sparse(select_values, 0, select_keys, 6);
        ret.wait();
        print_matrix("pull_sparse", select_values, 6, 2, select_keys); 
        ASSERT_EQ(select_values[1][1] > 0, true);
    }
    sleep(1);
    ret = client->flush();
    ret.wait();*/
    //ret = client->stop_server();
    //ret.wait();
}


TEST(downpour_ps_server, InitByConf) {
}

/* vim: set ts=4 sw=4 sts=4 tw=100 */
