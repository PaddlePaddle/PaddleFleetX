#pragma once

#include <memory>
#include <vector>
#include <unordered_map>
#include "ps_env.h"
#include "common/registerer.h"
#include "table/table.h"
#include "proto/ps.pb.h"
#include "butil/endpoint.h"
#include "google/protobuf/service.h"

namespace paddle {
namespace ps {
class PSServer {
public:
    PSServer() {}
    virtual ~PSServer() {}
    PSServer(PSServer&&) = delete;
    PSServer(const PSServer&) = delete;

    virtual int32_t configure(const PSParameter& config, 
        PSEnvironment& env, size_t server_rank) final {
        _config = config.server_param();
        _rank = server_rank;
        _environment = &env;
        //const auto& downpour_param = _config.downpour_server_param();
        const auto& downpour_param = _config.downpour_server_param();
        //for (size_t i = 0; i < downpour_param.downpour_table_param_size(); ++i) {
        for (size_t i = 0; i < downpour_param.downpour_table_param_size(); ++i) {
            auto* table = CREATE_CLASS(Table, 
                                       downpour_param.downpour_table_param(i).table_class());
                                       //downpour_param.downpour_table_param(i).table_class());
            //table->initialize(downpour_param.downpour_table_param(i), config.fs_client_param());
            table->initialize(downpour_param.downpour_table_param(i), config.fs_client_param());
            //_table_map[downpour_param.downpour_table_param(i).table_id()].reset(table);
            _table_map[downpour_param.downpour_table_param(i).table_id()].reset(table);
        }

        return initialize(); 
    }
    
    //return server_ip
    virtual std::string ip() {
        return butil::my_ip_cstr();
    }
    //return server_port
    virtual int32_t port() = 0;

    virtual uint64_t start() = 0;
    virtual int32_t stop() = 0;

    inline size_t rank() const {
        return _rank;
    }
    
    inline PSEnvironment* environment() {
        return _environment;
    }
    
    inline const ServerParameter* config() const {
        return &_config;
    }
    inline Table* table(size_t table_id) {
        auto itr = _table_map.find(table_id);
        if (itr != _table_map.end()) {
            return itr->second.get();
        }
        return NULL;
    }
    inline std::unordered_map<uint32_t, std::shared_ptr<Table>>* table() {
        return &_table_map;
    }

protected:
    virtual int32_t initialize() = 0;

protected:
    size_t _rank;
    ServerParameter _config;
    PSEnvironment* _environment;
    std::unordered_map<uint32_t, std::shared_ptr<Table>> _table_map;
};
REGISTER_REGISTERER(PSServer);

class PsBaseService : public PsService {
public:
    PsBaseService() : _rank(0), _server(NULL), _config(NULL) {}
    virtual ~PsBaseService() {}
    
    virtual int32_t configure(PSServer* server) {
        _server = server;
        _rank = _server->rank();
        _config = _server->config();
        return 0;
    }
    virtual void service(::google::protobuf::RpcController* controller,
                       const ::paddle::PsRequestMessage* request,
                       ::paddle::PsResponseMessage* response,
                       ::google::protobuf::Closure* done) override = 0;

    virtual void set_response_code(
        PsResponseMessage& response, int err_code, const char* err_msg) {
        response.set_err_msg(err_msg);
        response.set_err_code(err_code);
        LOG(WARNING) << "Resonse err_code:" << err_code << " msg:" << err_msg;
    }

    virtual int32_t initialize() = 0;
protected:
    size_t _rank;
    PSServer* _server;
    const ServerParameter* _config;
};
REGISTER_REGISTERER(PsBaseService);

class PSServerFactory {
public:
    static PSServer* create(const PSParameter& config);
};

}
}
