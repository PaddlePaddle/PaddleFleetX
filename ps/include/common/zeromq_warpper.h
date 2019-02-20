#pragma once

#include <vector>
#include <string>
#include "google/protobuf/stubs/callback.h"

namespace paddle {
namespace ps {

struct ZmqClientConfig {
    int32_t server_port;
    std::string server_ip;
    int32_t timeout_ms;
    int32_t retry_times;
};

struct ZmqServerConfig {
    uint32_t port;
    uint32_t thread_num;
    bool random_bind_port;
};


class ZmqMessage {
public:
    ZmqMessage() {}
    ~ZmqMessage() {}
    char * data() {
        return NULL;
    }
    size_t length() {
        return 0;
    }
};

class ZmqClosure : google::protobuf::Closure {
public:
    ZmqClosure() : _status(0), _request(NULL), _response(NULL) {}
    ~ZmqClosure() {}
    virtual void Run() = 0;
private:
    uint32_t _status;
    const ZmqMessage* _request;
    const ZmqMessage* _response;
};

class ZmqClient {
public:
    ZmqClient() {}
    virtual ~ZmqClient() {}
    ZmqClient(ZmqClient&&) = delete;
    ZmqClient(const ZmqClient&) = delete;

    int32_t initialize(const ZmqClientConfig& config);
     
    int32_t flush();    
    int32_t connect();
    int32_t close();  //断开连接
    //异步send，暂时只实现异步版本就行, data保证回调后才释放
    int32_t send(char* data, uint32_t data_size, ZmqClosure* closure);
};


class ZmqService {
public:
    virtual int32_t service(const ZmqMessage& request, ZmqMessage& response) = 0;
};

class ZmqServer {
public:
    ZmqServer() {}
    virtual ~ZmqServer() {}
    ZmqServer(ZmqServer&&) = delete;
    ZmqServer(const ZmqServer&) = delete;

    int initialize(const ZmqServerConfig& config);
    int start_server(ZmqService* service);
    inline const ZmqServerConfig& config() const {
        return _config;
    }
private:
    ZmqServerConfig _config;
};

}
}
