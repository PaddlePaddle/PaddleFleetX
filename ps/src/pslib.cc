/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#include <iostream>
#include <fcntl.h>
#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include "pslib.h"

using namespace std;

namespace paddle{
namespace distributed{
        /*
        paddle::PSParameter load_ps_parameter(const char* filename) {
            paddle::PSParameter param;
            int file_descriptor = open(filename, O_RDONLY);
            if (file_descriptor == -1){
                exit(-1);
            }
            google::protobuf::io::FileInputStream fileInput(file_descriptor);
            if (!google::protobuf::TextFormat::Parse(&fileInput, &param)) {
                //LOG(ERROR) << "FATAL: fail to parse " << filename;
                exit(-1);
            }
            close(file_descriptor);
            return param;
        }*/
        

        //int PSlib::init_and_config(const char* filename, uint64_t* host_sign_list, int node_num, int index) {
        int PSlib::init_server(const std::string& filename, int index) {

            //_ps_param = load_ps_parameter("parallel_config.prototxt");//TODO
            google::protobuf::TextFormat::ParseFromString(filename, &_ps_param);
            _ps_env = paddle::ps::PaddlePSEnvironment();
            int ret = 0;
            if (index % 2 == 0) {
                _server_ptr = std::shared_ptr<paddle::ps::PSServer>(paddle::ps::PSServerFactory::create(_ps_param));
                ret = _server_ptr->configure(_ps_param, _ps_env, index / 2);
                CHECK(ret == 0) << "failed to configure server";
            } 
            return ret;
        }
        int PSlib::init_worker(const std::string& filename, uint64_t* host_sign_list, int node_num, int index) {
            google::protobuf::TextFormat::ParseFromString(filename, &_ps_param);
            _ps_env = paddle::ps::PaddlePSEnvironment();
            _ps_env.set_ps_servers(host_sign_list, node_num);
            int ret = 0;
            if (index % 2 == 1) {
                _worker_ptr = std::shared_ptr<paddle::ps::PSClient>(paddle::ps::PSClientFactory::create(_ps_param));
                ret = _worker_ptr->configure(_ps_param, _ps_env, index / 2);
            }
            return ret;
        }
        int PSlib::gather_servers(uint64_t* host_sign_list, int node_num) {
            _ps_env.set_ps_servers(host_sign_list, node_num);
        }
        uint64_t PSlib::run_server() {
            return _server_ptr->start();
            //return 0;
        }
        int PSlib::stop_server() {
            auto stop_status = _worker_ptr->stop_server();
            stop_status.wait();
            return 0;
        }
        paddle::PSParameter* PSlib::get_param() {
            return &_ps_param;
        }
    }
}

