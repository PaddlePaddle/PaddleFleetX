#include <omp.h>
#include "table/table.h"
#include "communicate/downpour_ps_client.h"
#include "Eigen/Dense"

#include "common/thread_pool.h"
DEFINE_int32(pslib_push_dense_merge_limit, 12, "limit max push_dense local merge requests");
DEFINE_int32(pslib_push_sparse_merge_limit, 12, "limit max push_sparse local merge requests");
DEFINE_int32(pslib_async_push_dense_interval_ms, 10, "async push_dense to server interval");
DEFINE_int32(pslib_async_push_sparse_interval_ms, 10, "async push_sparse to server interval");
DEFINE_bool(pslib_scale_gradient_by_merge, false, "scale dense gradient when merged");
DEFINE_int32(pslib_communicate_compress_type, 0, "none:0 snappy:1 gzip:2 zlib:3 lz4:4");
DEFINE_int32(pslib_max_async_call_num, 13, "max task num in async_call_server");
DEFINE_int32(pslib_timeout_ms, 100000, "pslib request server timeout_ms");
DEFINE_int32(pslib_connect_timeout_ms, 10000, "pslib connect server timeout_ms");
DEFINE_string(pslib_connection_type, "pooled", "pslib connection_type[pooled:single]");

//#define pslib_debug_dense_compress

namespace paddle {
namespace ps {
    void DownpourPsClientService::service(
        ::google::protobuf::RpcController* controller, const ::paddle::PsRequestMessage* request,
        ::paddle::PsResponseMessage* response, ::google::protobuf::Closure* done) {
        brpc::ClosureGuard done_guard(done);
        int ret = _client->handle_client2client_msg(request->cmd_id(), request->client_id(), request->data());
        response->set_err_code(0);
        response->set_err_msg("");
        if (ret != 0) {
            response->set_err_code(-1);
            response->set_err_msg("handle_client2client_msg failed");
        }
    }

    //启动client端RpcService 用于数据互发等操作
    int32_t DownpourBrpcPsClient::start_client_service() {
        if (_service.configure(this, _client_id) != 0) {
            LOG(ERROR) << "service initialize failed, service_name:DownpourPsClientService";
            return -1;
        }
        _server.AddService(&_service, brpc::SERVER_DOESNT_OWN_SERVICE);
        brpc::ServerOptions options;
        int start_port = 8500;
        const static int max_port = 65535;
        options.num_threads = 8;
             
        if (_server.Start(butil::my_ip_cstr(), 
            brpc::PortRange(start_port,  max_port), &options) != 0) {
            LOG(ERROR) << "DownpourBrpcPsServer start failed";
            return -1;
        }
        _env->registe_ps_client(butil::my_ip_cstr(), _server.listen_address().port, _client_id);
        return 0;
    }

    int32_t DownpourBrpcPsClient::initialize() {
        _async_call_num = 0;
       
        brpc::ChannelOptions options;
        options.protocol = "baidu_std";
        options.timeout_ms = FLAGS_pslib_timeout_ms;
        options.connection_type = FLAGS_pslib_connection_type;
        options.connect_timeout_ms = FLAGS_pslib_connect_timeout_ms; 
        options.max_retry = 3;

        std::ostringstream os;
        std::string server_ip_port;
        std::string client_ip(butil::my_ip_cstr());
        
        //获取server列表，并连接
        std::vector<PSHost> server_list = _env->get_ps_servers();
        _server_channels.resize(server_list.size()); 
        for (size_t i = 0; i < server_list.size(); ++i) {
            server_ip_port.assign(server_list[i].ip.c_str());
            server_ip_port.append(":");
            server_ip_port.append(std::to_string(server_list[i].port));
            for (size_t j = 0; j < _server_channels[i].size(); ++j) {
                _server_channels[i][j].reset(new brpc::Channel());
                if (_server_channels[i][j]->Init(server_ip_port.c_str(), "", &options) != 0) {
                    LOG(ERROR) << "psclient connect to server:" << server_ip_port << " Failed!";
                    return -1;
                }
            }
            os << server_ip_port << ",";
        }
        
        //启动client探听接口, 并相互建立连接
        /*
        start_client_service();
        _env->gather_ps_clients();
        std::vector<PSHost> client_list = _env->get_ps_clients();
        _client_channels.resize(client_list.size()); 
        for (size_t i = 0; i < client_list.size(); ++i) {
            server_ip_port.assign(client_list[i].ip.c_str());
            server_ip_port.append(":");
            server_ip_port.append(std::to_string(client_list[i].port));
            _client_channels[i].reset(new brpc::Channel());
            if (_client_channels[i]->Init(server_ip_port.c_str(), "", &options) != 0) {
                LOG(ERROR) << "psclient connect to client:" << server_ip_port << " Failed!";
            }
            os << server_ip_port << ",";
        }
        */
        LOG(INFO) << "Client connect success:" << os.str();

        //异步push 请求队列初始化
        //const auto& work_param = _config.worker_param().downpour_worker_param();
        const auto& work_param = _config.worker_param().downpour_worker_param();
        //for (size_t i = 0; i < work_param.downpour_table_param_size(); ++i) {
        for (size_t i = 0; i < work_param.downpour_table_param_size(); ++i) {
            //auto type = work_param.downpour_table_param(i).type();
            auto type = work_param.downpour_table_param(i).type();
            //auto table_id = work_param.downpour_table_param(i).table_id();
            auto table_id = work_param.downpour_table_param(i).table_id();
            if (type == PS_DENSE_TABLE) {
                _push_dense_task_queue_map[table_id] = std::make_shared<DenseAsyncTaskQueue>();
            }
            if (type == PS_SPARSE_TABLE) {
                _push_sparse_task_queue_map[table_id] = std::make_shared<SparseAsyncTaskQueue>();
                _push_sparse_merge_count_map[table_id] = 0;
            }
        }

        //profiler
        auto& profiler = CostProfiler::instance();
        profiler.register_profiler("pslib_downpour_client_pull_dense");
        profiler.register_profiler("pslib_downpour_client_push_dense");
        profiler.register_profiler("pslib_downpour_client_push_dense_rpc");
        profiler.register_profiler("pslib_downpour_client_push_dense_parse");
        profiler.register_profiler("pslib_downpour_client_push_dense_merge");
        profiler.register_profiler("pslib_downpour_client_pull_sparse");
        profiler.register_profiler("pslib_downpour_client_pull_sparse_local");
        profiler.register_profiler("pslib_downpour_client_push_sparse");
        profiler.register_profiler("pslib_downpour_client_push_sparse_rpc");
        profiler.register_profiler("pslib_downpour_client_push_sparse_parse");
        profiler.register_profiler("pslib_downpour_client_push_sparse_merge");
        
        _running = true;
        _flushing = false;
        //启动异步push线程
        _async_push_sparse_thread = std::thread(
                std::bind(&DownpourBrpcPsClient::push_sparse_task_consume, this));
        _async_push_sparse_thread.detach();
        _async_push_dense_thread = std::thread(
                std::bind(&DownpourBrpcPsClient::push_dense_task_consume, this));
        _async_push_dense_thread.detach();

        return 0;
    }
    
    int DownpourBrpcClosure::check_response(size_t request_idx, int cmd_id) {
        if (FLAGS_pslib_is_debug) {
            LOG(INFO) <<  "resquest cmd_id:" << cmd_id 
                << ", cost_ms:" << _cntls[request_idx]->latency_us() / 1000
                << ", remote:" << butil::endpoint2str(_cntls[request_idx]->remote_side());
        }
        if (_cntls[request_idx]->Failed()) {
            LOG(ERROR) << "resquest cmd_id:" << cmd_id << " failed, "
                "err:" << _cntls[request_idx]->ErrorText();
            return -1;
        }
        if (_responses[request_idx].err_code() != 0) {
            LOG(ERROR) << "response ret bad, server_idx:" << request_idx
                << "cmd_id:" << cmd_id
                << " err_code:" << _responses[request_idx].err_code()
                << " err_msg:" << _responses[request_idx].err_msg();
            return -1;
        }
        return 0;
    }

    int DownpourBrpcClosure::check_save_response(size_t request_idx, int cmd_id) {
        uint32_t feasign_size = 0;
        if (FLAGS_pslib_is_debug) {
            LOG(INFO) <<  "resquest cmd_id:" << cmd_id 
                << ", cost_ms:" << _cntls[request_idx]->latency_us() / 1000
                << ", remote:" << butil::endpoint2str(_cntls[request_idx]->remote_side());
        }
        if (_cntls[request_idx]->Failed()) {
            LOG(ERROR) << "resquest cmd_id:" << cmd_id << " failed, "
                "err:" << _cntls[request_idx]->ErrorText();
            return -1;
        }
        feasign_size = _responses[request_idx].err_code();
        if (feasign_size < 0) {
            LOG(ERROR) << "response ret bad, server_idx:" << request_idx
                << "cmd_id:" << cmd_id
                << " err_code:" << _responses[request_idx].err_code()
                << " err_msg:" << _responses[request_idx].err_msg();
            return -1;
        }
        return feasign_size;
    }


    ::std::future<int32_t> DownpourBrpcPsClient::send_cmd(
        uint32_t table_id, int cmd_id, const std::vector<std::string>& params) {
        size_t request_call_num = _server_channels.size();
        DownpourBrpcClosure* closure = new DownpourBrpcClosure(request_call_num,
            [request_call_num, cmd_id](void* done) {
            int ret = 0;
            auto* closure = (DownpourBrpcClosure*)done;
            for (size_t i = 0; i < request_call_num; ++i) {
                if (closure->check_response(i, cmd_id) != 0) {
                    ret = -1;
                    break;
                }
            }
            closure->set_promise_value(ret);
        });
        auto promise = std::make_shared<std::promise<int32_t>>();
        closure->add_promise(promise);
        std::future<int> fut = promise->get_future();
        for (size_t i = 0; i < request_call_num; ++i) {
            closure->request(i)->set_cmd_id(cmd_id);
            closure->request(i)->set_table_id(table_id);
            closure->request(i)->set_client_id(_client_id);
            for (const auto& param : params) {
                closure->request(i)->add_params(param);
            }
            PsService_Stub rpc_stub(get_cmd_channel(i));
            closure->cntl(i)->set_timeout_ms(10800000);   //cmd msg don't limit timeout for save/load
            rpc_stub.service(closure->cntl(i), closure->request(i), closure->response(i), closure);
        }
        return fut;
    }
    
    ::std::future<int32_t> DownpourBrpcPsClient::send_save_cmd(
        uint32_t table_id, int cmd_id, const std::vector<std::string>& params) {
        size_t request_call_num = _server_channels.size();
        DownpourBrpcClosure* closure = new DownpourBrpcClosure(request_call_num,
            [request_call_num, cmd_id](void* done) {
            int ret = 0;
            uint32_t feasign_size = 0;
            auto* closure = (DownpourBrpcClosure*)done;
            for (size_t i = 0; i < request_call_num; ++i) {
                if (closure->check_save_response(i, cmd_id) < 0) {
                    ret = -1;
                    break;
                }
                feasign_size += closure->check_save_response(i, cmd_id);
            }
            if (ret == 0) {
                closure->set_promise_value(feasign_size);
            }
            else {
                closure->set_promise_value(ret);
            }
        });
        auto promise = std::make_shared<std::promise<int32_t>>();
        closure->add_promise(promise);
        std::future<int> fut = promise->get_future();
        for (size_t i = 0; i < request_call_num; ++i) {
            closure->request(i)->set_cmd_id(cmd_id);
            closure->request(i)->set_table_id(table_id);
            closure->request(i)->set_client_id(_client_id);
            for (const auto& param : params) {
                closure->request(i)->add_params(param);
            }
            PsService_Stub rpc_stub(get_cmd_channel(i));
            closure->cntl(i)->set_timeout_ms(10800000);   //cmd msg don't limit timeout for save/load
            rpc_stub.service(closure->cntl(i), closure->request(i), closure->response(i), closure);
        }
        return fut;
    }

    ::std::future<int32_t> DownpourBrpcPsClient::shrink(uint32_t table_id) {
        return send_cmd(table_id, PS_SHRINK_TABLE, {std::string("1")}); 
    }
    
    ::std::future<int32_t> DownpourBrpcPsClient::load(
        const std::string& epoch, const std::string& mode) {
        return send_cmd(-1, PS_LOAD_ALL_TABLE, {epoch, mode}); 
    }
    ::std::future<int32_t> DownpourBrpcPsClient::load(
        uint32_t table_id, const std::string& epoch, const std::string& mode) {
        return send_cmd(table_id, PS_LOAD_ONE_TABLE, {epoch, mode}); 
    }
    
    ::std::future<int32_t> DownpourBrpcPsClient::save(
        const std::string& epoch, const std::string& mode) {
        return send_save_cmd(-1, PS_SAVE_ALL_TABLE, {epoch, mode}); 
    }
    ::std::future<int32_t> DownpourBrpcPsClient::save(
        uint32_t table_id, const std::string& epoch, const std::string& mode) {
        return send_save_cmd(table_id, PS_SAVE_ONE_TABLE, {epoch, mode}); 
    }
    
    ::std::future<int32_t> DownpourBrpcPsClient::clear() {
        return send_cmd(-1, PS_CLEAR_ALL_TABLE, {}); 
    }
    ::std::future<int32_t> DownpourBrpcPsClient::clear(uint32_t table_id) {
        return send_cmd(table_id, PS_CLEAR_ONE_TABLE, {}); 
    }
    
    ::std::future<int32_t> DownpourBrpcPsClient::flush() {
        _flushing = true;
        std::promise<int> promise;
        ::std::future<int32_t> fut = promise.get_future();
        do {
            LOG(INFO) << "wait _async_call_num:" << _async_call_num;
            usleep(100000); //sleep 100ms wait async end
        }  while (_async_call_num > 0);
        promise.set_value(0);
        _flushing = false;
        return fut;
    }
    
    ::std::future<int32_t> DownpourBrpcPsClient::stop_server() {
        flush();
        _running = false;
        try {
            _async_push_dense_thread.join();
            _async_push_sparse_thread.join();
        } catch (...) {}
        return send_cmd(-1, PS_STOP_SERVER, {}); 
    }

    float regions_debug_sum(const Region* regions, size_t region_num, int use_idx = 1) {
        size_t dense_value_count = 0;
        float region_value_sum = 0.0;
        for (size_t i = 0; i < region_num; ++i) {
            float* region_data = (float*)regions[i].data;
            size_t value_num = regions[i].size / sizeof(float);
            for (size_t j = 0; j < value_num; ++j) {
                if (use_idx == 1) {
                    region_value_sum += region_data[j] * dense_value_count;
                } else {
                    region_value_sum += region_data[j];
                }
                ++dense_value_count;
            }
       }
       return region_value_sum;
    }

    ::std::future<int32_t> DownpourBrpcPsClient::pull_dense(
        Region* regions, size_t region_num, size_t table_id) {
        if (FLAGS_pslib_open_strict_check) {
            auto* accessor = table_accessor(table_id);
            CHECK(accessor != NULL) << "table not found, table_id:" << table_id;
            uint32_t region_size_total = 0;
            for (size_t i = 0; i < region_num; ++i) {
                region_size_total += regions[i].size;
            }
            CHECK(region_size_total == accessor->select_size() * accessor->fea_dim()) 
                << "regions size:"<< region_size_total 
                << " not equel to accessor select size:" << accessor->select_size() * accessor->fea_dim();
        }
        auto timer = std::make_shared<CostTimer>("pslib_downpour_client_pull_dense");
        auto* accessor = table_accessor(table_id);
        size_t request_call_num = _server_channels.size();
        uint32_t num_per_shard = dense_dim_per_shard(accessor->fea_dim(), request_call_num);
        //callback 将各shard结果，顺序填入region
        DownpourBrpcClosure* closure = new DownpourBrpcClosure(request_call_num,
            [request_call_num, num_per_shard, regions, region_num, accessor](void* done) {
            int ret = 0;
            size_t region_idx = 0;          //当前填充的region偏移
            size_t region_data_idx = 0;     //当前填充的region内data偏移
            auto* closure = (DownpourBrpcClosure*)done;
            size_t shard_data_size = num_per_shard * accessor->select_size();
            for (size_t i = 0; i < request_call_num; ++i) {
                if (closure->check_response(i, PS_PULL_DENSE_TABLE) != 0) {
                    ret = -1;
                    break;
                }
                auto& res_io_buffer = closure->cntl(i)->response_attachment();

                if (FLAGS_pslib_enable_pull_dense_compress) {
                    thread_local std::vector<float> data;
                    thread_local std::vector<Eigen::half> raw_data;
                    auto size = res_io_buffer.size() / sizeof(uint16_t);
                    raw_data.resize(size);
                    data.resize(size);

                    res_io_buffer.copy_to((void*)raw_data.data(), res_io_buffer.size(), 0);

                    typedef Eigen::Matrix<Eigen::half, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXFP16;
                    const Eigen::half* ptr = (const Eigen::half*)raw_data.data();
                    Eigen::Map<const MatrixXFP16> src_mat(ptr, 1, size);
                    Eigen::Map<Eigen::MatrixXf> mat(data.data(), 1, size);
                    mat << src_mat.template cast<float>();

                    auto shard_buffer_remain = size * sizeof(float);
                    if (shard_buffer_remain != shard_data_size) {
                        LOG(ERROR) << "expect res_size:" << shard_data_size << 
                            ", but size:" << shard_buffer_remain << ", ignore this response";
                        ret = -1;
                        break;
                    }
                    char* copy_ptr = (char*)data.data();
                    while (shard_buffer_remain > 0 && region_idx < region_num) {
                        auto& region = regions[region_idx];
                        if (region.size - region_data_idx >= shard_buffer_remain) {
                            memcpy((void*)(region.data + region_data_idx), copy_ptr, shard_buffer_remain);
                            copy_ptr += shard_buffer_remain;
                        } else if (region.size - region_data_idx == 0) {
                            ++region_idx;
                            region_data_idx = 0;
                        } else {
                            memcpy((void*)(region.data + region_data_idx), copy_ptr, region.size - region_data_idx);
                            ++region_idx;
                            region_data_idx = 0;
                            shard_buffer_remain -= (region.size - region_data_idx);
                        }
                    }
                } else {
                    butil::IOBufBytesIterator io_buffer_itr(res_io_buffer);
                    size_t shard_buffer_remain = res_io_buffer.size();
                    if (shard_buffer_remain != shard_data_size) {
                        LOG(ERROR) << "expect res_size:" << shard_data_size << 
                            ", but size:" << shard_buffer_remain << ", ignore this response";
                        ret = -1;
                        break;
                    }
                    while (shard_buffer_remain > 0 && region_idx < region_num) {
                        auto& region = regions[region_idx];
                        if (region.size - region_data_idx >= shard_buffer_remain) {
                            //region待填充空间 >= 分片buffer数据, 直接拷贝置入
                            io_buffer_itr.copy_and_forward(
                                (void*)(region.data + region_data_idx), shard_buffer_remain);
                            region_data_idx += shard_buffer_remain;
                            shard_buffer_remain = 0;
                        } else if (region.size - region_data_idx == 0) {
                            //region填满，切换到下一个region
                            ++region_idx;
                            region_data_idx = 0;
                        } else {
                            //region不足以容纳所有数据，则能放多少 拷贝多少
                            io_buffer_itr.copy_and_forward(
                                (void*)(region.data + region_data_idx), region.size - region_data_idx);
                            shard_buffer_remain -= (region.size - region_data_idx);
                            ++region_idx;
                            region_data_idx = 0;
                        }
                    }
                }
            }
            closure->set_promise_value(ret);
        });
        closure->add_timer(timer);
        auto promise = std::make_shared<std::promise<int32_t>>();
        closure->add_promise(promise);
        std::future<int> fut = promise->get_future();
        for (size_t i = 0; i < request_call_num; ++i) {
            closure->request(i)->set_cmd_id(PS_PULL_DENSE_TABLE);
            closure->request(i)->set_table_id(table_id);
            closure->request(i)->set_client_id(_client_id);
            closure->request(i)->add_params((char*)&num_per_shard, sizeof(num_per_shard));
            PsService_Stub rpc_stub(get_dense_channel(i));
            rpc_stub.service(closure->cntl(i), closure->request(i), closure->response(i), closure);
        }
        return fut;
    }

    DownpourBrpcClosure* DownpourBrpcPsClient::make_push_closure(int request_call_num, PsCmdID cmd) {
        return new DownpourBrpcClosure(request_call_num,
            [request_call_num, cmd](void* done) {
            auto* closure = (DownpourBrpcClosure*)done;
            for (size_t i = 0; i < request_call_num; ++i) {
                if (closure->check_response(i, cmd) != 0) {
                    break;
                }
            }
        });
    }

    ::std::future<int32_t> DownpourBrpcPsClient::push_dense_param(
        const Region* regions, size_t region_num, size_t table_id) {
        if (FLAGS_pslib_open_strict_check) {
            auto* accessor = table_accessor(table_id);
            CHECK(accessor != NULL) << "table not found, table_id:" << table_id;
            uint32_t region_size_total = 0;
            for (size_t i = 0; i < region_num; ++i) {
                region_size_total += regions[i].size;
            }
            CHECK(region_size_total == accessor->update_size() * accessor->fea_dim()) 
                << "regions size:"<< region_size_total 
                << " not equel to accessor update size:" << accessor->fea_dim();
            //LOG(WARNING) << "push_dense_param_check_sum:" << regions_debug_sum(regions, region_num);  
        }
        auto* accessor = table_accessor(table_id);
        size_t request_call_num = _server_channels.size();
        //1.拆分Region数据到shard中，后续多shard并行拷贝数据
        std::vector<std::vector<Region>> regions_partition(request_call_num);
        uint32_t num_per_shard = dense_dim_per_shard(accessor->fea_dim(), request_call_num);
        size_t shard_data_size = num_per_shard * accessor->update_size();
        size_t current_region_idx = 0;
        size_t current_region_data_idx = 0;
        for (size_t i = 0; i < request_call_num; ++i) {
            size_t shard_data_remain_size = shard_data_size;
            while (shard_data_remain_size > 0 && current_region_idx < region_num) {
                const auto& region = regions[current_region_idx];
                size_t region_remain_size = region.size - current_region_data_idx;
                if (shard_data_remain_size >= region_remain_size) {
                    regions_partition[i].push_back(
                        Region(region.data + current_region_data_idx, region_remain_size));
                    ++current_region_idx;
                    current_region_data_idx = 0;
                    shard_data_remain_size -= region_remain_size;
                } else {
                    regions_partition[i].push_back(
                        Region(region.data + current_region_data_idx, shard_data_remain_size));
                    current_region_data_idx += shard_data_remain_size;
                    shard_data_remain_size = 0;
                }
            }
        }

        DownpourBrpcClosure* closure = new DownpourBrpcClosure(request_call_num,
            [request_call_num](void* done) {
            int ret = 0;
            auto* closure = (DownpourBrpcClosure*)done;
            for (size_t i = 0; i < request_call_num; ++i) {
                if (closure->check_response(i, PS_PUSH_DENSE_PARAM) != 0) {
                    ret = -1;
                    break;
                }
            }
            closure->set_promise_value(ret);
        });
        auto promise = std::make_shared<std::promise<int32_t>>();
        closure->add_promise(promise);
        std::future<int> fut = promise->get_future();
        static const int REGION_ASSIGN_BUFFER_SIZE = 1024 * 10;
        static char region_assign_buffer[REGION_ASSIGN_BUFFER_SIZE]; //用于数据补齐
        //开始多shard并行拷贝&请求
        for (size_t i = 0; i < request_call_num; ++i) {
            closure->request(i)->set_cmd_id(PS_PUSH_DENSE_PARAM);
            closure->request(i)->set_table_id(table_id);
            closure->request(i)->set_client_id(_client_id);
            auto& request_buffer = closure->cntl(i)->request_attachment();
            request_buffer.append((void*)&num_per_shard, sizeof(uint32_t));
            auto& region_list = regions_partition[i];
            size_t fill_remain_size = shard_data_size; 
            for (auto& region : region_list) {
                fill_remain_size -= region.size;
                request_buffer.append((void*)region.data, region.size); 
            }
            //保证各分片数据对齐
            while (fill_remain_size > 0) {
                size_t fill_num = fill_remain_size > REGION_ASSIGN_BUFFER_SIZE ? 
                    REGION_ASSIGN_BUFFER_SIZE : fill_remain_size;
                request_buffer.append((void*)region_assign_buffer, fill_num);
                fill_remain_size -= fill_num;
            }
            PsService_Stub rpc_stub(get_dense_channel(i));
            rpc_stub.service(closure->cntl(i), closure->request(i), closure->response(i), closure);
        }
        return fut;
    }

    void DownpourBrpcPsClient::push_dense_compress_gradient(
            std::shared_ptr<DenseAsyncTask>& task, 
            float* total_send_data,
            size_t total_send_data_size,
            DownpourBrpcClosure* closure) {
        auto* accessor = table_accessor(task->table_id());
        size_t request_call_num = _server_channels.size();
        //将数据拷贝到请求buffer区
        auto timer = std::make_shared<CostTimer>("pslib_downpour_client_push_dense_rpc");
        closure->add_timer(timer);
        uint32_t num_per_shard = dense_dim_per_shard(accessor->fea_dim(), request_call_num);

        thread_local std::vector<Eigen::half> send_data;
        send_data.resize(total_send_data_size);

        Eigen::Map<Eigen::MatrixXf> mat(total_send_data, 1, total_send_data_size);

        typedef Eigen::Matrix<Eigen::half, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXUint16;
        Eigen::Map<MatrixXUint16> c_mat(send_data.data(), 1, total_send_data_size);
        c_mat << mat.template cast<Eigen::half>();

        for (size_t i = 0; i < request_call_num; ++i) {
            closure->request(i)->set_cmd_id(PS_PUSH_DENSE_TABLE);
            closure->request(i)->set_table_id(task->table_id());
            closure->request(i)->set_client_id(_client_id);
            auto* push_data = closure->request(i)->mutable_data();
            push_data->clear();
            push_data->resize(sizeof(uint32_t) + num_per_shard * sizeof(uint16_t));
            char* push_data_ptr = const_cast<char*>(push_data->data());
            char* ptr = push_data_ptr;
            memcpy(ptr, &num_per_shard, sizeof(uint32_t));
            ptr += sizeof(uint32_t);

            memcpy(ptr, send_data.data() + i * num_per_shard, num_per_shard * sizeof(uint16_t));
            closure->cntl(i)->set_request_compress_type(
                (brpc::CompressType)FLAGS_pslib_communicate_compress_type);
            PsService_Stub rpc_stub(get_dense_channel(i));
            rpc_stub.service(closure->cntl(i), closure->request(i), closure->response(i), closure);
        }
    }
    void DownpourBrpcPsClient::push_dense_raw_gradient(
            std::shared_ptr<DenseAsyncTask>& task, 
            float* total_send_data,
            size_t total_send_data_size,
            DownpourBrpcClosure* closure) {
        auto* accessor = table_accessor(task->table_id());
        size_t request_call_num = _server_channels.size();
        //将数据拷贝到请求buffer区
        auto timer = std::make_shared<CostTimer>("pslib_downpour_client_push_dense_rpc");
        closure->add_timer(timer);
        uint32_t num_per_shard = dense_dim_per_shard(accessor->fea_dim(), request_call_num);
        for (size_t i = 0; i < request_call_num; ++i) {
            closure->request(i)->set_cmd_id(PS_PUSH_DENSE_TABLE);
            closure->request(i)->set_table_id(task->table_id());
            closure->request(i)->set_client_id(_client_id);
            auto* push_data = closure->request(i)->mutable_data();
            push_data->clear();
            push_data->resize(sizeof(uint32_t) + num_per_shard * sizeof(float));
            char* push_data_ptr = const_cast<char*>(push_data->data());
            memcpy(push_data_ptr, &num_per_shard, sizeof(uint32_t));
            memcpy(push_data_ptr + sizeof(uint32_t), 
                total_send_data + i * num_per_shard, num_per_shard * sizeof(float));
            closure->cntl(i)->set_request_compress_type(
                (brpc::CompressType)FLAGS_pslib_communicate_compress_type);
            PsService_Stub rpc_stub(get_dense_channel(i));
            rpc_stub.service(closure->cntl(i), closure->request(i), closure->response(i), closure);
        }
    }
    
    void DownpourBrpcPsClient::push_dense_task_consume() {
        uint64_t merge_size = FLAGS_pslib_push_dense_merge_limit;
        static bool scale_gradient = FLAGS_pslib_scale_gradient_by_merge;
        ThreadPool<int> async_merge_dense_threads(10);
        while (_running) {
            auto async_start_time_ms = butil::gettimeofday_ms();
            for (auto& task_queue_itr : _push_dense_task_queue_map) {
                auto& task_queue = task_queue_itr.second;
                if (task_queue->size() <= merge_size) {
                    continue;
                }
                ++_async_call_num;
                std::shared_ptr<DenseAsyncTask> task(task_queue->pop());
                auto* accessor = table_accessor(task->table_id());
                //设置请求回调
                size_t request_call_num = _server_channels.size();

                DownpourBrpcClosure* closure = new DownpourBrpcClosure(request_call_num,
                        [this, request_call_num](void* done) {
                    int ret = 0;
                    auto* closure = (DownpourBrpcClosure*)done;
                    for (size_t i = 0; i < request_call_num; ++i) {
                        if (closure->check_response(i, PS_PUSH_DENSE_TABLE) != 0) {
                            ret = -1;
                            break;
                        }
                    }
                    closure->set_promise_value(ret);
                    --_async_call_num;
                });

                auto& total_send_data_vec = *(task->data());
                float* total_send_data = const_cast<float*>(total_send_data_vec.data());
                size_t total_send_data_size = total_send_data_vec.size();
                {
                    CostTimer merge_timer("pslib_downpour_client_push_dense_merge");
                    uint32_t merge_count = 0;
                    std::vector<std::future<int>> merge_status(merge_size);
                    while (!task_queue->empty() && merge_count < merge_size) {
                        auto* async_task = task_queue->pop();
                        closure->add_timer(async_task->timer());
                        closure->add_promise(async_task->promise());
                        merge_status[merge_count] = async_merge_dense_threads.AddTask(
                            [closure, accessor, &total_send_data, total_send_data_size, async_task]() -> int {
                            auto& tmp_task_vec = *(async_task->data());
                            const float* merge_data = tmp_task_vec.data();
                            accessor->merge(&total_send_data, &merge_data, total_send_data_size);
                            delete async_task;
                            return 0;
                        });
                        ++merge_count;
                    }
                    for (int i = 0; i < merge_count; ++i) {
                        merge_status[i].wait();
                    }
                    if (scale_gradient && merge_count > 1) {
                        Eigen::Map<Eigen::MatrixXf> mat(total_send_data, 1, total_send_data_size);
                        mat *= (1.0 / merge_count);
                    }


                }
                if (FLAGS_enable_dense_gradient_compress) {
                    push_dense_compress_gradient(task, total_send_data, total_send_data_size, closure);
                } else {
                    push_dense_raw_gradient(task, total_send_data, total_send_data_size, closure);
                }

            }
            auto wait_ms = FLAGS_pslib_async_push_dense_interval_ms - 
                (butil::gettimeofday_ms() - async_start_time_ms);
            if (wait_ms > 0) {
                usleep(wait_ms * 1000); 
            }
        }
    }
    
    ::std::future<int32_t> DownpourBrpcPsClient::push_dense(
        const Region* regions, size_t region_num, size_t table_id) {
        auto* accessor = table_accessor(table_id);
        if (FLAGS_pslib_open_strict_check) {
            CHECK(accessor != NULL) << "table not found, table_id:" << table_id;
            uint32_t region_size_total = 0;
            for (size_t i = 0; i < region_num; ++i) {
                region_size_total += regions[i].size;
            }
            CHECK(region_size_total == accessor->update_size() * accessor->fea_dim()) 
                << "regions size:"<< region_size_total 
                << " not equel to accessor update size:" << accessor->fea_dim();
            CHECK(_push_dense_task_queue_map.find(table_id) != _push_dense_task_queue_map.end())
                << "table_id:" << table_id << " is not a type of PS_DENSE_TABLE";
            //LOG(WARNING) << "push_dense_check_sum:" << regions_debug_sum(regions, region_num);  
            //LOG(WARNING) << "push_dense_raw_sum:" << regions_debug_sum(regions, region_num, 0);  
        }
        auto push_timer = std::make_shared<CostTimer>("pslib_downpour_client_push_dense");
        auto parse_timer = std::make_shared<CostTimer>("pslib_downpour_client_push_dense_parse");
        int push_dense_async_num = _push_dense_task_queue_map[table_id]->size();
        while (push_dense_async_num > FLAGS_pslib_max_async_call_num) {
            //LOG(INFO) << "Waiting for async_call_num comsume, task_num:" 
            //    << push_dense_async_num << ", max_task_limit:" << FLAGS_pslib_max_async_call_num;
            usleep(5000);//5ms
            push_dense_async_num = _push_dense_task_queue_map[table_id]->size();
        }
        auto dense_data = _dense_matrix_obj_pool.get();
        auto async_task = new DenseAsyncTask(dense_data, table_id, push_timer);
        size_t request_call_num = _server_channels.size();
        uint32_t num_per_shard = dense_dim_per_shard(accessor->fea_dim(), request_call_num);
        
        //将region数据拷贝到转置矩阵中
        async_task->data()->resize(
            num_per_shard * request_call_num * accessor->update_dim());
        float* data = async_task->data()->data();
        size_t data_size = async_task->data()->size();
        uint32_t pos = 0;
        for (size_t i = 0; i < region_num; ++i) {
            uint32_t data_num = regions[i].size / sizeof(float);
            CHECK(pos + data_num <= data_size) << "invalid dense size, cur pos[" << pos << "]"
                << " data_num[" << data_num << "] size[" << data_size << "]";
            const float* region_data = (const float*)(regions[i].data); 
            memcpy(data + pos, region_data, regions[i].size);
            pos += data_num;
        }
        std::future<int> fut = async_task->get_future();
        _push_dense_task_queue_map[table_id]->push(std::move(async_task));
        return fut;
    }

    ::std::future<int32_t> DownpourBrpcPsClient::pull_sparse(
        float** select_values, size_t table_id, const uint64_t* keys, size_t num) {
        auto timer = std::make_shared<CostTimer>("pslib_downpour_client_pull_sparse");
        auto local_timer = std::make_shared<CostTimer>("pslib_downpour_client_pull_sparse_local");
        //将key拆分到各shard请求，并记录原始对应value指针
        auto shard_sorted_kv_list = std::make_shared<std::vector<
            std::vector<std::pair<uint64_t, float*>>>>();
        auto* accessor = table_accessor(table_id);
        size_t request_call_num = _server_channels.size();
        shard_sorted_kv_list->resize(request_call_num);
        for (size_t i = 0; i < num; ++i) {
            size_t shard_id = SparseTable::get_sparse_shard(
                FLAGS_pslib_sparse_table_shard_num, request_call_num, keys[i]);
            shard_sorted_kv_list->at(shard_id).push_back({keys[i], select_values[i]});
        }
        DownpourBrpcClosure* closure = new DownpourBrpcClosure(request_call_num,
            [shard_sorted_kv_list, accessor](void* done) {
            int ret = 0;
            auto* closure = (DownpourBrpcClosure*)done;
            for (size_t i = 0; i < shard_sorted_kv_list->size(); ++i) {

                if (closure->check_response(i, PS_PULL_SPARSE_TABLE) != 0) {
                    ret = -1;
                    break;
                }
                //将response返回填充到原请求buffer中
                auto& request_kv_list = shard_sorted_kv_list->at(i);
                auto& res_io_buffer = closure->cntl(i)->response_attachment();
                butil::IOBufBytesIterator io_buffer_itr(res_io_buffer);
                uint64_t last_key = UINT64_MAX;
                float* last_value_data = NULL;
                size_t value_size = accessor->select_size();
                for (size_t kv_idx = 0; kv_idx < request_kv_list.size(); ++kv_idx) {
                    auto* kv_pair = &(request_kv_list[kv_idx]);
                    if (kv_pair->first == last_key) {
                        memcpy((void*)kv_pair->second, (void*)last_value_data, value_size);
                    } else {
                        last_key = kv_pair->first;
                        last_value_data = kv_pair->second;
                        if (value_size != io_buffer_itr.copy_and_forward((
                            void*)(last_value_data), value_size)) {
                            LOG(WARNING) << "res data is lack or not in format";
                            ret = -1;
                            break;
                        }
                    }
                }
            }
            closure->set_promise_value(ret);
        });
        closure->add_timer(timer);
        auto promise = std::make_shared<std::promise<int32_t>>();
        closure->add_promise(promise);
        std::future<int> fut = promise->get_future();
        
        for (size_t i = 0; i < request_call_num; ++i) {
            //按key排序&去重
            auto& sorted_kv_list = shard_sorted_kv_list->at(i);
            std::sort(sorted_kv_list.begin(), sorted_kv_list.end(), 
                [](const std::pair<uint64_t, float*>& k1, const std::pair<uint64_t, float*>& k2) {
                return k1.first < k2.first;
            });
            uint64_t last_key = UINT64_MAX;
            uint32_t kv_request_count = 0;
            size_t sorted_kv_size = sorted_kv_list.size();
            auto& request_buffer = closure->cntl(i)->request_attachment();
            for (size_t kv_idx = 0; kv_idx < sorted_kv_size; ++kv_idx) {
                ++kv_request_count;
                last_key = sorted_kv_list[kv_idx].first;
                request_buffer.append((void*)&last_key, sizeof(uint64_t));
                while (kv_idx < sorted_kv_size - 1 && last_key == sorted_kv_list[kv_idx + 1].first) {
                    ++kv_idx;
                }
            }
            if (kv_request_count == 0) {
                closure->Run();     //无请求,则直接回调,保证request计数
            } else {
                closure->request(i)->set_cmd_id(PS_PULL_SPARSE_TABLE);
                closure->request(i)->set_table_id(table_id);
                closure->request(i)->set_client_id(_client_id);
                closure->request(i)->add_params((char*)&kv_request_count, sizeof(uint32_t));
                PsService_Stub rpc_stub(get_cmd_channel(i));
                closure->cntl(i)->set_log_id(butil::gettimeofday_ms());
                rpc_stub.service(closure->cntl(i), closure->request(i), closure->response(i), closure);
            }
        }
        return fut;
    }
    
    template<class T>
    struct array_deleter {
        void operator()(T* &x) const { delete[] x; }
    };

    void sparse_local_merge(ValueAccessor* accessor, float* merge_data, const float* another_data) {
        size_t col_num = accessor->update_size() / sizeof(float);
        float* merge_data_shell[col_num];
        const float* another_data_shell[col_num];
        for (int i = 0; i < col_num; ++i) {
            merge_data_shell[i] = merge_data + i;
            another_data_shell[i] = another_data + i;
        }
        accessor->merge(merge_data_shell, another_data_shell, 1);
    }
    
    int DownpourBrpcPsClient::push_sparse_async_shard_merge(
            std::vector<std::shared_ptr<SparseAsyncTask>>& task_list, 
            std::vector<int>& request_kv_num, int table_id, int shard_idx, ValueAccessor* accessor) {
        size_t merged_kv_count = 0;
        uint64_t min_key = UINT64_MAX;
        //std::vector<int> min_key_task_ids;
        //std::vector<int> task_kv_idx(task_list.size(), 0);

        //将队列数据汇总排序，本地merge  最终数据汇总到task_list[0]
        auto sorted_kv_list = 
            std::vector<std::pair<uint64_t, const float*>>();

        for (int i = 1; i < task_list.size(); ++i) {
            auto& key_list = task_list[i]->data()->shared_data[shard_idx].key_list;
            auto& value_list = task_list[i]->data()->shared_data[shard_idx].value_list;
            
            for (int i = 0; i < key_list.size(); ++i) {
                char* task_data_ptr = const_cast<char*>(value_list[i].data());
                sorted_kv_list.push_back({key_list[i], (float*)task_data_ptr});
            }
        }
      
        //按key排序&去重
        std::sort(sorted_kv_list.begin(), sorted_kv_list.end(), 
                [](const std::pair<uint64_t, const float*>& k1, 
                const std::pair<uint64_t, const float*>& k2) {
                return k1.first < k2.first;
        });

        auto& async_task = task_list[0];
        size_t sorted_kv_size = sorted_kv_list.size();
        auto& shard_kv_data = async_task->data()->shared_data[shard_idx];
        shard_kv_data.key_list.resize(sorted_kv_size);
        shard_kv_data.value_list.resize(sorted_kv_size);
        
        //将去重后数据写入分shard包
        if (sorted_kv_size == 0) {
            shard_kv_data.kv_num = 0;
            return 0;
        }

        //去重 本地merge
        uint64_t last_key = sorted_kv_list[0].first;
        const float* last_value_data = sorted_kv_list[0].second;
        uint32_t value_size = accessor->update_size();
        float* last_merge_data = NULL;
        std::shared_ptr<char> merger_buffer(
            new char[value_size], array_deleter<char>());
        for (size_t kv_idx = 1; kv_idx < sorted_kv_size; ++kv_idx) {
            while (kv_idx < sorted_kv_size && last_key == sorted_kv_list[kv_idx].first) {
                if (last_merge_data == NULL) {
                    last_merge_data = (float*)merger_buffer.get();
                    memcpy(last_merge_data, last_value_data, value_size);
                }
                sparse_local_merge(accessor, last_merge_data, sorted_kv_list[kv_idx].second);
                ++kv_idx;
            }
            if (last_merge_data != NULL) {
                shard_kv_data.value_list[merged_kv_count].assign(
                    (const char*)last_merge_data, value_size);
                last_merge_data = NULL; 
            } else {
                shard_kv_data.value_list[merged_kv_count].assign(
                    (const char*)sorted_kv_list[kv_idx - 1].second, value_size);
            }
            shard_kv_data.key_list[merged_kv_count++] = last_key;
            if (kv_idx < sorted_kv_size) {
                last_key = sorted_kv_list[kv_idx].first;
                last_value_data = sorted_kv_list[kv_idx].second;
            }
            if (kv_idx == sorted_kv_size - 1) {
                shard_kv_data.value_list[merged_kv_count].assign(
                    (const char*)last_value_data, value_size);
                shard_kv_data.key_list[merged_kv_count++] = last_key;
            }
        }
        shard_kv_data.kv_num = merged_kv_count;
        return 0;

        //遍历每个sparseShard, shard内KV 不是有序且去重的，需要排序多路归并
/*        while (true) {
            for (int i = 0; i < task_list.size(); ++i) {
                auto& task_shard_data = task_list[i]->data()->shared_data[shard_idx];
                if (task_kv_idx[i] >= task_shard_data.kv_num) {
                    continue;
                }
                uint64_t key = task_shard_data.key_list[task_kv_idx[i]];
                if (key < min_key) {
                    min_key = key;
                    min_key_task_ids.clear();
                    min_key_task_ids.push_back(i);
                } else if (key == min_key) {
                    min_key_task_ids.push_back(i);
                }
            }
            //所有kv节点遍历完成
            if (min_key_task_ids.size() == 0) {
                break;           
            }
            float* data_for_merge = NULL;
            for (int i = 0; i < min_key_task_ids.size(); ++i) {
                int task_id = min_key_task_ids[i];
                auto& task_shard_data = task_list[task_id]->data()->shared_data[shard_idx];
                char* task_data_ptr = const_cast<char*>(task_shard_data.
                value_list[task_kv_idx[task_id]].data());
                if (data_for_merge == NULL) {
                    data_for_merge = (float*)task_data_ptr;
                } else {
                    sparse_local_merge(accessor, data_for_merge, (float*)task_data_ptr);
                }
                ++task_kv_idx[task_id];
            }
            merged_key_list[merged_kv_count] = min_key;
            merged_value_list[merged_kv_count] = data_for_merge;
            min_key = UINT64_MAX;
            min_key_task_ids.clear();
            ++merged_kv_count;
        } */
    }

    int DownpourBrpcPsClient::push_sparse_async_shard_push(
            std::vector<std::shared_ptr<SparseAsyncTask>>& task_list, 
            std::vector<int>& request_kv_num, int table_id, int shard_idx,
            DownpourBrpcClosure* closure, ValueAccessor* accessor) {

        push_sparse_async_shard_merge(task_list, request_kv_num, table_id, shard_idx, accessor);
        size_t merged_kv_count = task_list[0]->data()->shared_data[shard_idx].key_list.size();

        //thread_local std::vector<uint64_t> merged_key_list;
        //thread_local std::vector<float*> merged_value_list;
        //thread_local std::vector<std::string> merged_value_list;
        auto& merged_key_list = task_list[0]->data()->shared_data[shard_idx].key_list;
        auto& merged_value_list = task_list[0]->data()->shared_data[shard_idx].value_list;
         
        //merged_key_list.resize(request_kv_num[shard_idx]);
        //merged_value_list.resize(request_kv_num[shard_idx]);

        //发送RPC请求
        auto* push_request = closure->request(shard_idx);
        push_request->set_cmd_id(PS_PUSH_SPARSE_TABLE);
        push_request->set_table_id(table_id);
        push_request->set_client_id(_client_id);
        push_request->add_params((char*)&merged_kv_count, sizeof(uint32_t));
        auto* push_data = push_request->mutable_data();
        push_data->resize(merged_kv_count * (sizeof(uint64_t) + accessor->update_size()));
        char* push_data_ptr = const_cast<char*>(push_data->data());
        memcpy(push_data_ptr, merged_key_list.data(), merged_kv_count * sizeof(uint64_t));
        push_data_ptr += merged_kv_count * sizeof(uint64_t);
        for (int i = 0; i < merged_kv_count; ++i) {
            const char* task_data_ptr = merged_value_list[i].data();

            memcpy(push_data_ptr, (float*)task_data_ptr, accessor->update_size());
            push_data_ptr += accessor->update_size();
        }
        PsService_Stub rpc_stub(get_sparse_channel(shard_idx));
        closure->cntl(shard_idx)->set_request_compress_type(
             (brpc::CompressType)FLAGS_pslib_communicate_compress_type);
        rpc_stub.service(closure->cntl(shard_idx), 
        closure->request(shard_idx), closure->response(shard_idx), closure);
        _push_sparse_merge_count_map[table_id] = 0;
        return 0;
    }

    //改为每次都聚合。维护一个merge_count判断发送时机。
    void DownpourBrpcPsClient::push_sparse_task_consume() {
        uint64_t merge_size = FLAGS_pslib_push_sparse_merge_limit;
        std::vector<std::shared_ptr<SparseAsyncTask>> task_list;
        //task_list.reserve();
        size_t request_call_num = _server_channels.size();
        ThreadPool<int> async_push_sparse_shard_threads(10);
        while (_running) {
            auto async_start_time_ms = butil::gettimeofday_ms();
            //所有sparseTable的pushTask 进行处理
            for (auto& push_sparse_task_itr : _push_sparse_task_queue_map) {
                auto table_id = push_sparse_task_itr.first;
                auto* accessor = table_accessor(table_id);
                auto& task_queue = push_sparse_task_itr.second;
                if (merge_size > 0 && (task_queue->size() <= 1 && _flushing == false)) {
                    continue;
                }
                ++_async_call_num;
                
                int merge_count = 0;
                task_list.clear();
                int cur_meger_size = task_queue->size();

                //task_list[0] 为一个空SparseAsyncTask, 分shard异步merge结果存入此结构。
                auto sparse_task_data = _sparse_push_obj_pool.get();
                sparse_task_data->shared_data.resize(request_call_num);
                auto push_timer = std::make_shared<CostTimer>("pslib_downpour_client_push_sparse");

                auto async_task = new SparseAsyncTask(sparse_task_data, table_id, push_timer);
      
                task_list.reserve(cur_meger_size + 1);

                task_list.push_back(std::move(std::shared_ptr<SparseAsyncTask>(async_task)));

                while (!task_queue->empty() && merge_count < cur_meger_size) {
                    ++merge_count;
                    task_list.push_back(std::shared_ptr<SparseAsyncTask>(task_queue->pop()));
                }
                
                _push_sparse_merge_count_map[table_id] += merge_count;

                //达到或大于 merge_size发送, 发送过程中
                std::vector<int> request_kv_num(request_call_num, 0);

                if (_push_sparse_merge_count_map[table_id] >= merge_size || _flushing == true) {
                    DownpourBrpcClosure* closure = new DownpourBrpcClosure(request_call_num,
                        [this, request_call_num](void* done) {
                        int ret = 0;
                        auto* closure = (DownpourBrpcClosure*)done;
                        for (size_t i = 0; i < request_call_num; ++i) {
                            if (closure->check_response(i, PS_PUSH_SPARSE_TABLE) != 0) {
                                ret = -1;
                                break;
                            }
                        }
                        closure->set_promise_value(ret);
                        --_async_call_num;
                    });
                    
                    for_each(task_list.begin() + 1, task_list.end(), 
                        [&request_kv_num, request_call_num, closure] (std::shared_ptr<SparseAsyncTask>& task) {
                        //for (int i = 0; i < request_call_num; ++i) {
                        //    request_kv_num[i] += task->data()->shared_data[i].kv_num;
                        //}
                        closure->add_timer(task->timer());
                        closure->add_promise(task->promise());
                    });

                    CostTimer merge_timer("pslib_downpour_client_push_sparse_merge");
                    auto rpc_timer = std::make_shared<CostTimer>("pslib_downpour_client_push_sparse_rpc");
                    closure->add_timer(rpc_timer);
                    
                    std::vector<std::future<int>> merge_status(request_call_num);
                    for (int shard_idx = 0; shard_idx < request_call_num; ++shard_idx) {
                        merge_status[shard_idx] = async_push_sparse_shard_threads.AddTask(
                            std::bind(&DownpourBrpcPsClient::push_sparse_async_shard_push, this,
                            task_list, request_kv_num, table_id, shard_idx, closure, accessor));
                    }
                    for (int shard_idx = 0; shard_idx < request_call_num; ++shard_idx) {
                       merge_status[shard_idx].wait();
                    }
                    task_list.clear();
                    _push_sparse_merge_count_map[table_id] = 0;
                }

                //未达到阈值 只做多路归并
                else {
                    std::vector<std::future<int>> merge_status(request_call_num);
                    for (int shard_idx = 0; shard_idx < request_call_num; ++shard_idx) {
                        merge_status[shard_idx] = async_push_sparse_shard_threads.AddTask(
                            std::bind(&DownpourBrpcPsClient::push_sparse_async_shard_merge, this,
                            task_list, request_kv_num, table_id, shard_idx, accessor));
                    }
                    for (int shard_idx = 0; shard_idx < request_call_num; ++shard_idx) {
                       merge_status[shard_idx].wait();
                    }
                    //meger到task_list[0]
                   
                    auto async_task = new SparseAsyncTask(*(task_list[0].get()));

                    task_queue->push(std::move(async_task));
                    //task_list.reserve(cur_meger_size + 1);
                    //task_list.push_back(std::move(std::shared_ptr<SparseAsyncTask>(async_task)));
                    //std::move(std::shared_ptr<SparseAsyncTask>(async_task))
                    --_async_call_num;
                    task_list.clear();
                }
            }
            auto wait_ms = FLAGS_pslib_async_push_sparse_interval_ms - 
                (butil::gettimeofday_ms() - async_start_time_ms);
            if (wait_ms > 0) {
                usleep(wait_ms * 1000); 
            }
        }
    }
    ::std::future<int32_t> DownpourBrpcPsClient::push_sparse(
        size_t table_id, const uint64_t* keys, const float** update_values, size_t num) {
        CostTimer parse_timer("pslib_downpour_client_push_sparse_parse");
        auto push_timer = std::make_shared<CostTimer>("pslib_downpour_client_push_sparse");
        int push_sparse_async_num = _push_sparse_task_queue_map[table_id]->size();
        while (push_sparse_async_num > FLAGS_pslib_max_async_call_num) {
            LOG(INFO) << "Waiting for async_call_num comsume, task_num:" 
                << push_sparse_async_num << ", max_task_limit:" << FLAGS_pslib_max_async_call_num;
            usleep(5000);//5ms
            push_sparse_async_num = _push_sparse_task_queue_map[table_id]->size();
        }
        //将key拆分到各shard请求，并记录原始对应value指针
        auto shard_sorted_kv_list = std::make_shared<std::vector<
            std::vector<std::pair<uint64_t, const float*>>>>();
        auto* accessor = table_accessor(table_id);
        size_t request_call_num = _server_channels.size();
        shard_sorted_kv_list->resize(request_call_num);
        for (size_t i = 0; i < num; ++i) {
            size_t shard_id = SparseTable::get_sparse_shard(
                FLAGS_pslib_sparse_table_shard_num, request_call_num, keys[i]);

            shard_sorted_kv_list->at(shard_id).push_back({keys[i], update_values[i]});
        }
        auto sparse_task_data = _sparse_push_obj_pool.get();
        sparse_task_data->shared_data.resize(request_call_num);
        auto async_task = new SparseAsyncTask(sparse_task_data, table_id, push_timer);
        
        for (size_t i = 0; i < request_call_num; ++i) {
            //按key排序&去重
            auto& sorted_kv_list = shard_sorted_kv_list->at(i);
            /*std::sort(sorted_kv_list.begin(), sorted_kv_list.end(), 
                [](const std::pair<uint64_t, const float*>& k1, 
                const std::pair<uint64_t, const float*>& k2) {
                return k1.first < k2.first;
            });*/
            size_t sorted_kv_size = sorted_kv_list.size();
            auto& shard_kv_data = async_task->data()->shared_data[i];
            shard_kv_data.key_list.resize(sorted_kv_size);
            shard_kv_data.value_list.resize(sorted_kv_size);
            
            //将去重后数据写入分shard包
            if (sorted_kv_size == 0) {
                shard_kv_data.kv_num = 0;
                continue;
            }

            //直接插入排序去重下移到_push_sparse_task_queue_map
            uint32_t value_size = accessor->update_size();
            for (size_t kv_idx = 0; kv_idx < sorted_kv_size; ++kv_idx) {
                shard_kv_data.key_list[kv_idx] = sorted_kv_list[kv_idx].first;
                //shard_kv_data.value_list[kv_idx].assign(
                //        (const char*)sorted_kv_list[kv_idx].second, value_size);
                //shard_kv_data.value_list[kv_idx] = sorted_kv_list[kv_idx].second;
                shard_kv_data.value_list[kv_idx].assign(
                    (const char*)sorted_kv_list[kv_idx].second, value_size);
            }

            /*
            uint64_t last_key = sorted_kv_list[0].first;
            const float* last_value_data = sorted_kv_list[0].second;
            uint32_t value_size = accessor->update_size();
            size_t merged_kv_count = 0;
            float* last_merge_data = NULL;
            std::shared_ptr<char> merger_buffer(
                new char[value_size], array_deleter<char>());
            for (size_t kv_idx = 1; kv_idx < sorted_kv_size; ++kv_idx) {
                while (kv_idx < sorted_kv_size && last_key == sorted_kv_list[kv_idx].first) {
                    if (last_merge_data == NULL) {
                        last_merge_data = (float*)merger_buffer.get();
                        memcpy(last_merge_data, last_value_data, value_size);
                    }
                    sparse_local_merge(accessor, last_merge_data, sorted_kv_list[kv_idx].second);
                    ++kv_idx;
                }
                if (last_merge_data != NULL) {
                    shard_kv_data.value_list[merged_kv_count].assign(
                        (const char*)last_merge_data, value_size);
                    last_merge_data = NULL; 
                } else {
                    shard_kv_data.value_list[merged_kv_count].assign(
                        (const char*)sorted_kv_list[kv_idx - 1].second, value_size);
                }
                shard_kv_data.key_list[merged_kv_count++] = last_key;
                if (kv_idx < sorted_kv_size) {
                    last_key = sorted_kv_list[kv_idx].first;
                    last_value_data = sorted_kv_list[kv_idx].second;
                }
                if (kv_idx == sorted_kv_size - 1) {
                    shard_kv_data.value_list[merged_kv_count].assign(
                        (const char*)last_value_data, value_size);
                    shard_kv_data.key_list[merged_kv_count++] = last_key;
                }
            }
            shard_kv_data.kv_num = merged_kv_count;
            */
        }
        std::future<int> fut = async_task->get_future();
        _push_sparse_task_queue_map[table_id]->push(std::move(async_task));
        return fut;
    }
    
    ::std::future<int32_t> DownpourBrpcPsClient::send_client2client_msg(
        int msg_type, int to_client_id, const std::string& msg) {
        auto promise = std::make_shared<std::promise<int32_t>>();
        std::future<int> fut = promise->get_future();
        if (to_client_id >= _client_channels.size()) {
            LOG(FATAL) << "to_client_id is out of range clients, which size is " << _client_channels.size();
            promise->set_value(-1);
            return fut;
        }
        auto* closure = new DownpourBrpcClosure(1, [msg_type](void* done) {
            auto* closure = (DownpourBrpcClosure*)done;
            int32_t ret = closure->check_response(0, msg_type + 1000);
            closure->set_promise_value(ret);
        });
        closure->add_promise(promise);
        closure->request(0)->set_cmd_id(msg_type);
        closure->request(0)->set_client_id(_client_id);
        closure->request(0)->set_data(msg);
        PsService_Stub rpc_stub(_client_channels[to_client_id].get());
        rpc_stub.service(closure->cntl(0), closure->request(0), closure->response(0), closure);
        return fut;
    }
}
}
