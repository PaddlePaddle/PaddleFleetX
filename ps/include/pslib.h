/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include "communicate/ps_server.h"
#include "communicate/ps_client.h"

namespace paddle {
namespace distributed {

    class PSlib {
    public:
        explicit PSlib() {};
        virtual ~PSlib(){};

        virtual int init_server(const std::string& filename, int index);
        virtual int init_worker(const std::string& filename, uint64_t* host_sign_list, int node_num, int index);
        virtual uint64_t run_server();
        virtual int stop_server();
        virtual int gather_servers(uint64_t* host_sign_list, int node_num);
        std::shared_ptr<paddle::ps::PSServer> _server_ptr;  // pointer to server
        std::shared_ptr<paddle::ps::PSClient> _worker_ptr;  // pointer to worker
        virtual paddle::PSParameter* get_param();
    private:
        paddle::PSParameter _ps_param;
        paddle::ps::PaddlePSEnvironment _ps_env;
    };
    
}  // namespace distributed
}  // namespace paddle
