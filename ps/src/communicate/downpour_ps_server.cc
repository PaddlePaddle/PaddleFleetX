#include <thread>
#include "butil/endpoint.h"
#include "communicate/downpour_ps_server.h"
#include "Eigen/Dense"

DEFINE_int32(pslib_server_thread_num, 500, "thread_num for pslib_server in service");

namespace paddle {
namespace ps {

    int32_t DownpourBrpcPsServer::initialize() {
        //auto& service_config = _config.downpour_server_param().service_param();
        auto& service_config = _config.downpour_server_param().service_param();
        if (!service_config.has_service_class()) {
            LOG(ERROR) << "miss service_class in ServerServiceParameter";
            return -1;
        }
        auto* service = CREATE_CLASS(PsBaseService, service_config.service_class());
        if (service == NULL) {
            LOG(ERROR) << "service is unregisteg, serice_name:" << service_config.service_class();
            return -1;
        }

        _service.reset(service);
        if (service->configure(this) != 0
            || service->initialize() != 0) {
            LOG(ERROR) << "service initialize failed, service_name:" << service_config.service_class();
            return -1;
        }
        if (_server.AddService(service, brpc::SERVER_DOESNT_OWN_SERVICE) != 0) {
            LOG(ERROR) << "service add to brpc failed, service:" << service_config.service_class();
            return -1;
        }
        return 0;
    }
    
    uint64_t DownpourBrpcPsServer::start() {
        brpc::ServerOptions options;
        int start_port = 8000;
        const static int max_port = 65535;
        options.num_threads = FLAGS_pslib_server_thread_num;
             
        if (_server.Start(butil::my_ip_cstr(), 
            brpc::PortRange(start_port,  max_port), &options) != 0) {
            LOG(ERROR) << "DownpourBrpcPsServer start failed";
            return 0;
        }
        _environment->registe_ps_server(ip(), port(), _rank);
        PSHost host;
        host.ip = ip();
        host.port = port();
        host.rank = _rank;
        return host.serialize_to_uint64();
        //return 0;
    }
    
    int32_t DownpourBrpcPsServer::port() {
        return _server.listen_address().port;
    }

    int32_t DownpourPsService::initialize() {
        _is_initialize_shard_info = false;
        _service_handler_map[PS_STOP_SERVER] = &DownpourPsService::stop_server;
        _service_handler_map[PS_PULL_DENSE_TABLE] = &DownpourPsService::pull_dense;
        _service_handler_map[PS_PUSH_DENSE_TABLE] = &DownpourPsService::push_dense;
        _service_handler_map[PS_PULL_SPARSE_TABLE] = &DownpourPsService::pull_sparse;
        _service_handler_map[PS_PUSH_SPARSE_TABLE] = &DownpourPsService::push_sparse;
        _service_handler_map[PS_SAVE_ONE_TABLE] = &DownpourPsService::save_one_table;
        _service_handler_map[PS_SAVE_ALL_TABLE] = &DownpourPsService::save_all_table;
        _service_handler_map[PS_SHRINK_TABLE] = &DownpourPsService::shrink_table;
        _service_handler_map[PS_LOAD_ONE_TABLE] = &DownpourPsService::load_one_table;
        _service_handler_map[PS_LOAD_ALL_TABLE] = &DownpourPsService::load_all_table;
        _service_handler_map[PS_CLEAR_ONE_TABLE] = &DownpourPsService::clear_one_table;
        _service_handler_map[PS_CLEAR_ALL_TABLE] = &DownpourPsService::clear_all_table;
        _service_handler_map[PS_PUSH_DENSE_PARAM] = &DownpourPsService::push_dense_param;
        auto& profiler = CostProfiler::instance();
        profiler.register_profiler("pslib_downpour_server_pull_dense");
        profiler.register_profiler("pslib_downpour_server_push_dense");
        profiler.register_profiler("pslib_downpour_server_pull_sparse");
        profiler.register_profiler("pslib_downpour_server_push_sparse");
        return 0;
    }

    #define CHECK_TABLE_EXIST(table, request, response)                 \
        if (table == NULL) {                                            \
            std::string err_msg("table not found with table_id:");      \
            err_msg.append(std::to_string(request.table_id()));         \
            set_response_code(response, -1, err_msg.c_str());           \
            return -1;                                                  \
        }


    int32_t DownpourPsService::initialize_shard_info() {
        if (!_is_initialize_shard_info) {
            std::lock_guard<std::mutex> guard(_initialize_shard_mutex);
            if (_is_initialize_shard_info) {
                return 0;
            }
            size_t shard_num = _server->environment()->get_ps_servers().size();
            auto& table_map = *(_server->table());
            for (auto itr : table_map) {
                itr.second->set_shard(_rank, shard_num);
            }
            _is_initialize_shard_info = true;
        }
        return 0;
    }

    void DownpourPsService::service(google::protobuf::RpcController* cntl_base,
        const PsRequestMessage* request, PsResponseMessage* response, google::protobuf::Closure* done) {
        brpc::ClosureGuard done_guard(done);
        std::string log_label("ReceiveCmd-");
        if (!request->has_table_id()) {
            set_response_code(*response, -1, "PsRequestMessage.tabel_id is required");
            return;
        }
        
        //shard初始化,server启动后才可从env获取到server_list的shard信息
        initialize_shard_info();
        
        response->set_err_code(0);
        response->set_err_msg("");
        auto* table = _server->table(request->table_id());
        brpc::Controller* cntl = static_cast<brpc::Controller*>(cntl_base);
        auto itr = _service_handler_map.find(request->cmd_id());
        if (itr == _service_handler_map.end()) {
            std::string err_msg("undefined cmd_id, should match PsCmdID in ps.proto, cmd_id:");
            err_msg.append(std::to_string(request->cmd_id()));
            set_response_code(*response, -1, err_msg.c_str());
            return;
        }
        serviceHandlerFunc handler_func = itr->second;
        int service_ret = (this->*handler_func)(table, *request, *response, cntl);
        if (service_ret != 0) {
            response->set_err_code(service_ret);
            response->set_err_msg("server internal error");
        }
    }
    
    int32_t DownpourPsService::pull_dense(Table* table,
        const PsRequestMessage& request, PsResponseMessage& response, brpc::Controller* cntl) {
        CHECK_TABLE_EXIST(table, request, response)
        if (request.params_size() < 1) {
            set_response_code(response, -1, 
            "PsRequestMessage.datas is requeired at least 1 for num of dense");
            return 0;
        }
        CostTimer timer("pslib_downpour_server_pull_dense");
        uint32_t num = *(const uint32_t *)request.params(0).c_str();
        if (num < 0) {
            set_response_code(response, -1, "PsRequestMessage.datas[0] is invalid, num must >= 0");
            return 0;
        }
        auto res_data = table->pull_dense(num);
        if (FLAGS_pslib_enable_pull_dense_compress) {
            Eigen::Map<Eigen::MatrixXf> src_mat((float*)res_data->data(), 1, res_data->size());
            thread_local std::vector<Eigen::half> tmp;
            tmp.resize(res_data->size());
            typedef Eigen::Matrix<Eigen::half, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXFP16;
            Eigen::Map<MatrixXFP16> dst_mat(tmp.data(), 1, tmp.size());
            dst_mat << src_mat.template cast<Eigen::half>();
            cntl->response_attachment().append((char*)tmp.data(), tmp.size() * sizeof(uint16_t));
        } else {
            cntl->response_attachment().append(
                    (char*)res_data->data(), res_data->size() * sizeof(float));
        }
        return 0;
    }

    int32_t DownpourPsService::push_dense_param(Table* table,
        const PsRequestMessage& request, PsResponseMessage& response, brpc::Controller* cntl) {
        CHECK_TABLE_EXIST(table, request, response)
        thread_local std::string push_buffer;
        auto& req_io_buffer = cntl->request_attachment();
        auto req_buffer_size = req_io_buffer.size();
        if (req_buffer_size < 1) {
            set_response_code(response, -1, "req attachment is empty");
            return 0;
        }
        push_buffer.resize(0);
        push_buffer.reserve(req_buffer_size);
        const char* data = (const char*)cntl->request_attachment().
            fetch(const_cast<char*>(push_buffer.data()), req_buffer_size);

        /* 
        Attachment Content:
        |--num--|---valuesData---|
        |--4B---|----------------|
        */
        uint32_t num = *(const uint32_t*)data;
        const float* values = (const float*)(data + sizeof(uint32_t));
        if (table->push_dense_param(values, num) != 0) {
            set_response_code(response, -1, "push_dense_param failed");
        }
        return 0;
    }
    
    int32_t DownpourPsService::push_dense(Table* table,
        const PsRequestMessage& request, PsResponseMessage& response, brpc::Controller* cntl) {
        CHECK_TABLE_EXIST(table, request, response)
        auto req_buffer_size = request.data().size(); 
        if (req_buffer_size < 1) {
            //set_response_code(response, 0, "push dense data is empty");
            return 0;
        }

        if (FLAGS_enable_dense_gradient_compress) {
            /*
             * |---num--|--valuesData--|
             * |---4B---|----2B*num----|
             */
            int offset = 0;
            uint32_t num = *(const uint32_t*)(request.data().data() + offset);
            offset += sizeof(uint32_t);
            const Eigen::half* values = (const Eigen::half*)(request.data().data() + offset);

            typedef Eigen::Matrix<Eigen::half, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXUint16;
            Eigen::Map<const MatrixXUint16> mat(values, 1, num);
            _ori_values.resize(num);
            Eigen::Map<Eigen::MatrixXf> ori_mat(_ori_values.data(), 1, num);
            ori_mat << mat.template cast<float>();
            if (table->push_dense(_ori_values.data(), num) != 0) {
                set_response_code(response, -1, "push dense with compress gradient failed");
            }
        } else {
            /* 
            Push Content:
            |--num--|---valuesData---|
            |--4B---|----------------|
            */
            uint32_t num = *(const uint32_t*)(request.data().data());
            const float* values = (const float*)(request.data().data() + sizeof(uint32_t));
            if (table->push_dense(values, num) != 0) {
                set_response_code(response, -1, "push_dense failed");
            }
        }
        return 0;
    }

    int32_t DownpourPsService::pull_sparse(Table* table,
        const PsRequestMessage& request, PsResponseMessage& response, brpc::Controller* cntl) {
        CHECK_TABLE_EXIST(table, request, response)
        thread_local std::string push_sparse_request_buffer;
        auto& req_io_buffer = cntl->request_attachment();
        auto req_buffer_size = req_io_buffer.size();
        if (req_buffer_size < 1) {
            set_response_code(response, -1, "req attachment is empty");
            return 0;
        }
        if (request.params_size() < 1) {
            set_response_code(response, -1, 
            "PsRequestMessage.params is requeired at least 1 for num of sparse_key");
            return 0;
        }
        CostTimer timer("pslib_downpour_server_pull_sparse");
        uint32_t num = *(uint32_t*)(request.params(0).c_str());
        push_sparse_request_buffer.resize(0);
        push_sparse_request_buffer.reserve(req_buffer_size);
        const char* data = (const char*)cntl->request_attachment().
            fetch(const_cast<char*>(push_sparse_request_buffer.data()), req_buffer_size);
        /* 
        Attachment Content:
        |---keysData---|
        |---8*{num}B---|
        */
        const uint64_t* keys = (const uint64_t*)data;
        auto res_data = table->pull_sparse(keys, num);
        cntl->response_attachment().append(
            (char*)res_data->data(), res_data->size() * sizeof(float));
        return 0;
    }
    
    int32_t DownpourPsService::push_sparse(Table* table,
        const PsRequestMessage& request, PsResponseMessage& response, brpc::Controller* cntl) {
        CHECK_TABLE_EXIST(table, request, response)
        auto& push_data = request.data();
        if (push_data.size() < 1) {
            //set_response_code(response, 0, "push sparse data is empty");
            return 0;
        }
        if (request.params_size() < 1) {
            set_response_code(response, -1, 
            "PsRequestMessage.params is requeired at least 1 for num of sparse_key");
            return 0;
        }
        CostTimer timer("pslib_downpour_server_push_sparse");
        uint32_t num = *(uint32_t*)(request.params(0).c_str());
        /* 
        Push Content:
        |---keysData---|---valuesData---|
        |---8*{num}B---|----------------|
        */
        const uint64_t* keys = (const uint64_t*)push_data.data();
        const float* values = (const float*)(push_data.data() + sizeof(uint64_t) * num);
        if (table->push_sparse(keys, values, num) != 0) {
            set_response_code(response, -1, "push_sparse error");
        }
        return 0;
    }
    
    int32_t DownpourPsService::load_one_table(Table* table,
        const PsRequestMessage& request, PsResponseMessage& response, brpc::Controller* cntl) {
        CHECK_TABLE_EXIST(table, request, response)
        if (request.params_size() < 2) {
            set_response_code(response, -1, 
                "PsRequestMessage.datas is requeired at least 2 for path & load_param");
            return -1;
        }
        if (table->load(request.params(0), request.params(1)) != 0) {
            set_response_code(response, -1, "table load failed");
            return -1;
        }
        return 0;
    }

    int32_t DownpourPsService::load_all_table(Table* table,
        const PsRequestMessage& request, PsResponseMessage& response, brpc::Controller* cntl) {
        auto& table_map = *(_server->table());
        for (auto& itr : table_map) {
            if (load_one_table(itr.second.get(), request, response, cntl) != 0) {
                LOG(ERROR) << "load table[" << itr.first << "] failed";
                return -1;
            }
        }
        return 0;
    }

    int32_t DownpourPsService::save_one_table(Table* table,
        const PsRequestMessage& request, PsResponseMessage& response, brpc::Controller* cntl) {
        CHECK_TABLE_EXIST(table, request, response)
        if (request.params_size() < 2) {
            set_response_code(response, -1, "PsRequestMessage.datas is requeired at least 2, path&mode");
            return -1;
        }
        table->flush();

        int32_t feasign_size = 0;
        feasign_size = table->save(request.params(0), request.params(1));
        if (feasign_size < 0) {
            set_response_code(response, -1, "table save failed");
            return -1;
        }
        return feasign_size;
    }

    int32_t DownpourPsService::save_all_table(Table* table,
        const PsRequestMessage& request, PsResponseMessage& response, brpc::Controller* cntl) {
        auto& table_map = *(_server->table());
        int32_t all_feasign_size = 0;
        int32_t feasign_size = 0;

        for (auto& itr : table_map) {
            feasign_size = save_one_table(itr.second.get(), request, response, cntl);
            if (feasign_size < 0) {
                LOG(ERROR) << "save table[" << itr.first << "] failed";
                return -1;
            }
        }
        return 0;
    }
    
    int32_t DownpourPsService::shrink_table(Table* table,
        const PsRequestMessage& request, PsResponseMessage& response, brpc::Controller* cntl) {
        CHECK_TABLE_EXIST(table, request, response)
        table->flush();
        if (table->shrink() != 0) {
            set_response_code(response, -1, "table shrink failed");
        }
        return 0;
    }

    int32_t DownpourPsService::clear_one_table(Table* table,
        const PsRequestMessage& request, PsResponseMessage& response, brpc::Controller* cntl) {
        CHECK_TABLE_EXIST(table, request, response)
        table->flush();
        table->clear();
        return 0;
    }

    int32_t DownpourPsService::clear_all_table(Table* table,
        const PsRequestMessage& request, PsResponseMessage& response, brpc::Controller* cntl) {
        auto& table_map = *(_server->table());
        for (auto& itr : table_map) {
            if (clear_one_table(itr.second.get(), request, response, cntl) != 0) {
                return -1;
            }
        }
        return 0;
    }
    
    int32_t DownpourPsService::stop_server(Table* table,
        const PsRequestMessage& request, PsResponseMessage& response, brpc::Controller* cntl) {
        auto* p_server = _server;
        std::thread t_stop([p_server](){
            p_server->stop();
            LOG(INFO) << "Server Stoped";
        });
        t_stop.detach();
        return 0;
    }

}
}
