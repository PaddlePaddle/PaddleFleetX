#pragma once

#include <memory>
#include <vector>
#include "ps_server.h"
#include "brpc/server.h"

namespace paddle {
namespace ps {

class DownpourBrpcPsServer : public PSServer {
public:
    DownpourBrpcPsServer() {}
    virtual ~DownpourBrpcPsServer() {}
    virtual uint64_t start();
    virtual int32_t stop() {
        _server.Join();
        _server.Stop(1000);
        return 0;//TODO 
    }
    virtual int32_t port();
    
private:
    virtual int32_t initialize();
    
    brpc::Server _server;
    std::shared_ptr<PsBaseService> _service;
};

class DownpourPsService;

typedef int32_t (DownpourPsService::*serviceHandlerFunc)(Table* table, 
    const PsRequestMessage& request, PsResponseMessage& response, brpc::Controller* cntl);

class DownpourPsService : public PsBaseService {
public:
    virtual int32_t initialize() override;
    
    virtual void service(::google::protobuf::RpcController* controller,
                       const ::paddle::PsRequestMessage* request,
                       ::paddle::PsResponseMessage* response,
                       ::google::protobuf::Closure* done) override;
private:
    int32_t initialize_shard_info();
    int32_t pull_dense(Table* table,
        const PsRequestMessage& request, PsResponseMessage& response, brpc::Controller* cntl);
    int32_t push_dense(Table* table,
        const PsRequestMessage& request, PsResponseMessage& response, brpc::Controller* cntl);
    int32_t push_dense_param(Table* table,
        const PsRequestMessage& request, PsResponseMessage& response, brpc::Controller* cntl);
    int32_t pull_sparse(Table* table,
        const PsRequestMessage& request, PsResponseMessage& response, brpc::Controller* cntl);
    int32_t push_sparse(Table* table,
        const PsRequestMessage& request, PsResponseMessage& response, brpc::Controller* cntl);
    int32_t load_one_table(Table* table,
        const PsRequestMessage& request, PsResponseMessage& response, brpc::Controller* cntl);
    int32_t load_all_table(Table* table,
        const PsRequestMessage& request, PsResponseMessage& response, brpc::Controller* cntl);
    int32_t save_one_table(Table* table,
        const PsRequestMessage& request, PsResponseMessage& response, brpc::Controller* cntl);
    int32_t save_all_table(Table* table,
        const PsRequestMessage& request, PsResponseMessage& response, brpc::Controller* cntl);
    int32_t shrink_table(Table* table,
        const PsRequestMessage& request, PsResponseMessage& response, brpc::Controller* cntl);
    int32_t clear_one_table(Table* table,
        const PsRequestMessage& request, PsResponseMessage& response, brpc::Controller* cntl);
    int32_t clear_all_table(Table* table,
        const PsRequestMessage& request, PsResponseMessage& response, brpc::Controller* cntl);
    int32_t stop_server(Table* table,
        const PsRequestMessage& request, PsResponseMessage& response, brpc::Controller* cntl);

    bool _is_initialize_shard_info;
    std::mutex _initialize_shard_mutex;
    std::unordered_map<int32_t, serviceHandlerFunc> _service_handler_map;

    std::vector<float>  _ori_values;
};


}
}
