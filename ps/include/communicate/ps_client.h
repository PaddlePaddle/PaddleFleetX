#pragma once

#include <memory>
#include <future>
#include <vector>
#include <unordered_map>
#include "ps_env.h"
#include "proto/ps.pb.h"
#include "table/accessor.h"
#include "common/timer.h"
#include "common/pool.h"

namespace paddle {
namespace ps {

class PSClient {
public:
    PSClient() {}
    virtual ~PSClient() {}
    PSClient(PSClient&&) = delete;
    PSClient(const PSClient&) = delete;

    virtual int32_t configure(
        const PSParameter& config, PSEnvironment& _env, size_t client_id) final; 
    
    // 触发table数据退场
    virtual ::std::future<int32_t> shrink(uint32_t table_id) = 0;
    
    // 全量table进行数据load
    virtual ::std::future<int32_t> load(const std::string& epoch, const std::string& mode) = 0;
    // 指定table数据load
    virtual ::std::future<int32_t> load(uint32_t table_id, 
        const std::string& epoch, const std::string& mode) = 0;
    
    // 全量table数据save  value_accessor根据mode，可能有不同的save条件
    virtual ::std::future<int32_t> save(
        const std::string& epoch, const std::string& mode) = 0;
    // 指定table数据save  value_accessor根据mode，可能有不同的save条件
    virtual ::std::future<int32_t> save(
        uint32_t table_id, const std::string& epoch, const std::string& mode) = 0;

    //清空table数据
    virtual ::std::future<int32_t> clear() = 0;
    virtual ::std::future<int32_t> clear(uint32_t table_id) = 0;
     
    // pull dense的参数部分，并分块填充到本地网络参数中
    // start和num用于拉取部分参数
    // future结束前keys和values缓冲区不能再次使用
    // client将values按照区块拆包后送交多个sender
    // sender聚集同一区块的请求，累计多个填充buffer
    // server将参数区块中配置的某一维提取返回
    // 返回数据解包后填充到累计的多个buffer中
    virtual ::std::future<int32_t> pull_dense(
        Region* regions, size_t region_num, size_t table_id) = 0;
    
    // push dense的梯度，上传到server进行更新
    // start和num用于更新部分参数
    // future结束前keys和values缓冲区不能再次使用
    // client将values按照区块拆包后送交多个sender
    // sender聚集同一区块的请求，第一份拷贝赋值
    // 后续的使用_accessor.merge函数聚合
    // server使用update函数进行参数更新
    virtual ::std::future<int32_t> push_dense(
        const Region* regions, size_t region_num, size_t table_id) = 0;
    
    // firstly push dense param for parameter server
    // this is neccessary because dense weight initialized in trainer on cold start
    virtual ::std::future<int32_t> push_dense_param(
        const Region* regions, size_t region_num, size_t table_id) = 0;

    // 使用keys进行pull请求，结果填充values
    // keys和values的个数均为num个，每个value占用select_size空间
    // future结束前keys和values缓冲区不能再次使用
    // 整合多个线程请求的keys，聚集并分散发送到server
    // 返回结果后，遍历buffer并对values赋值
    virtual ::std::future<int32_t> pull_sparse(
        float** select_values, size_t table_id, const uint64_t* keys, size_t num) = 0;
    
    // 使用keys和values进行push请求
    // keys和values的个数均为num个，每个value占用update_size空间
    // future结束前keys和values缓冲区不能再次使用
    // 整合多个线程请求的keys，聚集并分散发送到server
    // 发送前对keys和values按照key排序，创建迭代器
    // 之后调用_accessor.merge(buffer, iterator)填充buffer
    virtual ::std::future<int32_t> push_sparse(
        size_t table_id, const uint64_t* keys, const float** update_values, size_t num) = 0;
    
    // 确保所有积攒中的请求都发起发送
    virtual ::std::future<int32_t> flush() = 0;
    //server优雅退出
    virtual ::std::future<int32_t> stop_server() = 0;

    //client to client, 消息发送
    virtual ::std::future<int32_t> send_client2client_msg(int msg_type, int to_client_id, const std::string& msg) {
        LOG(FATAL) << "Did not implement";
        std::promise<int32_t> promise;
        std::future<int> fut = promise.get_future();
        promise.set_value(-1);
        return fut;
    }
    //client2client消息处理，std::function<int32_t (int, int, const std::string&) -> ret (msg_type, from_client_id, msg)
    typedef std::function<int32_t (int, int, const std::string&)> MsgHandlerFunc;
    virtual int registe_client2client_msg_handler(int msg_type, MsgHandlerFunc handler) {
        _msg_handler_map[msg_type] = handler;
        return 0;
    } 
    virtual int handle_client2client_msg(int msg_type, int from_client_id, const std::string& msg) {
        auto itr = _msg_handler_map.find(msg_type);
        if (itr == _msg_handler_map.end()) {
            LOG(WARNING) << "unknown client2client_msg type:" << msg_type;
            return -1;
        }
        return itr->second(msg_type, from_client_id, msg);
    }
protected:
    virtual int32_t initialize() = 0;

    virtual ValueAccessor* table_accessor(size_t table_id) {
        auto itr = _table_accessors.find(table_id);
        if (itr == _table_accessors.end()) {
            return NULL;
        }
        return itr->second.get();
    }
    size_t _client_id;
    PSParameter _config;
    PSEnvironment* _env;
    std::unordered_map<uint32_t, std::shared_ptr<ValueAccessor>> _table_accessors; 
    std::unordered_map<int32_t, MsgHandlerFunc> _msg_handler_map; //处理client2client消息
};
REGISTER_REGISTERER(PSClient);

typedef std::function<void(void*)> PSClientCallBack;
class PSClientClosure : public google::protobuf::Closure {
public:
    PSClientClosure(PSClientCallBack callback) : _callback(callback) {}
    virtual ~PSClientClosure() {}
    virtual void set_promise_value(int value) {
        for (auto& promise : _promises) {
            promise->set_value(value);
        }
    }

    void add_promise(std::shared_ptr<std::promise<int32_t>>& promise) {
        _promises.push_back(promise);
    }
    
    void add_timer(std::shared_ptr<CostTimer>& timer) {
        _timers.push_back(timer);
    }
protected:
    PSClientCallBack _callback;
    std::vector<std::shared_ptr<CostTimer>> _timers;
    std::vector<std::shared_ptr<std::promise<int32_t>>> _promises;
};

template <class T>
class AsyncRequestTask {
public:
    AsyncRequestTask() : _promise(std::make_shared<std::promise<int32_t>>()){}
    AsyncRequestTask(T& data, size_t table_id, std::shared_ptr<CostTimer>& timer) : 
        _table_id(table_id), _timer(timer), 
        _promise(std::make_shared<std::promise<int32_t>>()) {
        _data = std::move(data);
    }

    AsyncRequestTask(AsyncRequestTask& data) : 
    _table_id(data.table_id()), _timer(data.timer()), 
    _promise(data.promise()) {
        _data = std::move(data.data());
    }
  
    ~AsyncRequestTask() {}

    inline T& data() {
        return _data;
    }
    inline size_t table_id() {
        return _table_id;
    }
    inline std::shared_ptr<CostTimer>& timer() {
        return _timer;
    }
    inline std::future<int32_t> get_future() {
        return _promise->get_future();
    }
    inline std::shared_ptr<std::promise<int32_t>>& promise() {
        return _promise;
    }
private:
    T _data;
    size_t _table_id;
    std::shared_ptr<CostTimer> _timer;
    std::shared_ptr<std::promise<int32_t>> _promise;
};

class PSClientFactory {
public:
    static PSClient* create(const PSParameter& config);
};

}
}
