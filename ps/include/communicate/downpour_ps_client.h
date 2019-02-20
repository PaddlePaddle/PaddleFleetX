#pragma once
#include "ps_client.h"
#include "brpc/channel.h"
#include "brpc/controller.h"
#include "common/thread_queue.h"
#include "common/channel.h"
#include "brpc/server.h"

namespace paddle {
namespace ps {

class DownpourPsClientService : public PsService {
public:
    DownpourPsClientService() {}
    virtual ~DownpourPsClientService() {}
    
    virtual int32_t configure(PSClient* client, size_t rank_id) {
        _client = client;
        _rank = rank_id;
        return 0;
    }
    virtual void service(
        ::google::protobuf::RpcController* controller, const ::paddle::PsRequestMessage* request,
        ::paddle::PsResponseMessage* response, ::google::protobuf::Closure* done) override;
protected:
    size_t _rank;
    PSClient* _client;
};

class DownpourBrpcClosure : public PSClientClosure {
public:
    DownpourBrpcClosure(size_t num, PSClientCallBack callback) : PSClientClosure(callback) {
        _waiting_num = num;

        _cntls.resize(num);
        _requests.resize(num);
        _responses.resize(num);
        for (size_t i = 0; i < num; ++i) {
            _cntls[i].reset(new brpc::Controller());
        }
    }
    virtual ~DownpourBrpcClosure() {
    }
    virtual void Run() override {
        if (_waiting_num.fetch_sub(1) == 1) {
            _callback(this);
            delete this;
        }
    }
    PsRequestMessage* request(size_t i) {
        return &_requests[i];
    }
    PsResponseMessage* response(size_t i) {
        return &_responses[i];
    }
    brpc::Controller* cntl(size_t i) {
        return _cntls[i].get();
    }
    int check_response(size_t request_idx, int cmd_id);
    int check_save_response(size_t request_idx, int cmd_id);
private:
    std::atomic<int32_t> _waiting_num;
    std::vector<PsRequestMessage> _requests;  //TODO cache it
    std::vector<PsResponseMessage> _responses; //TODO cache it
    std::vector<std::shared_ptr<brpc::Controller>> _cntls;
};

struct SharedSparsePushData {
    size_t kv_num;
    std::vector<uint64_t> key_list;
    std::vector<std::string> value_list;
};
struct SparsePushTaskData {
    std::vector<SharedSparsePushData> shared_data;   //sparse数据按key hash分片
};
typedef ObjectPool<SparsePushTaskData> SparsePushObjPool;
typedef SparsePushObjPool::PooledObject SparsePushPooledObj;

typedef ObjectPool<std::vector<float>> VectorObjPool;
typedef VectorObjPool::PooledObject VectorPooledObj;

typedef ObjectPool<std::vector<uint16_t>> Uint16VecObjPool;
typedef Uint16VecObjPool::PooledObject VPooledObj;

class DownpourBrpcPsClient : public PSClient {
public:
    DownpourBrpcPsClient() {}
    virtual ~DownpourBrpcPsClient() {
        _running = false;
        try {
            _async_push_dense_thread.join();
            _async_push_sparse_thread.join();
        } catch (...) {}
    }
    
    // 触发table数据退场
    virtual ::std::future<int32_t> shrink(uint32_t table_id) override;
    // 全量table进行数据load
    virtual ::std::future<int32_t> load(const std::string& epoch, const std::string& mode) override;
    // 指定table数据load
    virtual ::std::future<int32_t> load(uint32_t table_id, 
        const std::string& epoch, const std::string& mode) override;
    
    // 全量table数据save  value_accessor根据mode，可能有不同的save条件
    virtual ::std::future<int32_t> save(
        const std::string& epoch, const std::string& mode) override;
    // 指定table数据save  value_accessor根据mode，可能有不同的save条件
    virtual ::std::future<int32_t> save(
        uint32_t table_id, const std::string& epoch, const std::string& mode) override;
    
    //清空table数据
    virtual ::std::future<int32_t> clear() override;
    virtual ::std::future<int32_t> clear(uint32_t table_id) override;
    
    //server优雅退出
    virtual ::std::future<int32_t> stop_server() override;
     
    // pull dense的参数部分，并分块填充到本地网络参数中
    // start和num用于拉取部分参数
    // future结束前keys和values缓冲区不能再次使用
    // client将values按照区块拆包后送交多个sender
    // sender聚集同一区块的请求，累计多个填充buffer
    // server将参数区块中配置的某一维提取返回
    // 返回数据解包后填充到累计的多个buffer中
    virtual ::std::future<int32_t> pull_dense(
        Region* regions, size_t region_num, size_t table_id);
    
    // push dense的梯度，上传到server进行更新
    // start和num用于更新部分参数
    // future结束前keys和values缓冲区不能再次使用
    // client将values按照区块拆包后送交多个sender
    // sender聚集同一区块的请求，第一份拷贝赋值
    // 后续的使用_accessor.merge函数聚合
    // server使用update函数进行参数更新
    virtual ::std::future<int32_t> push_dense(
        const Region* regions, size_t region_num, size_t table_id);
    void push_dense_task_consume();
    
    virtual ::std::future<int32_t> push_dense_param(
        const Region* regions, size_t region_num, size_t table_id);

    // 使用keys进行pull请求，结果填充values
    // keys和values的个数均为num个，每个value占用select_size空间
    // future结束前keys和values缓冲区不能再次使用
    // 整合多个线程请求的keys，聚集并分散发送到server
    // 返回结果后，遍历buffer并对values赋值
    virtual ::std::future<int32_t> pull_sparse(
        float** select_values, size_t table_id, const uint64_t* keys, size_t num);
    
    // 使用keys和values进行push请求
    // keys和values的个数均为num个，每个value占用update_size空间
    // future结束前keys和values缓冲区不能再次使用
    // 整合多个线程请求的keys，聚集并分散发送到server
    // 发送前对keys和values按照key排序，创建迭代器
    // 之后调用_accessor.merge(buffer, iterator)填充buffer
    virtual ::std::future<int32_t> push_sparse(
        size_t table_id, const uint64_t* keys, const float** update_values, size_t num);
    void push_sparse_task_consume();
    
    // 确保所有积攒中的请求都发送完成
    virtual ::std::future<int32_t> flush();
    
    //client to client, 消息发送
    virtual ::std::future<int32_t> send_client2client_msg(int msg_type, int to_client_id, const std::string& msg) override;
private:
    virtual int32_t initialize() override;
    
    //计算每个shard 对 dense的存储量
    inline uint32_t dense_dim_per_shard(uint32_t dense_dim_total, uint32_t shard_num) {
        return dense_dim_total / shard_num + 1;
    }
    
    ::std::future<int32_t> send_cmd(uint32_t table_id, 
        int cmd_id, const std::vector<std::string>& param);

    ::std::future<int32_t> send_save_cmd(uint32_t table_id, 
        int cmd_id, const std::vector<std::string>& param);

    inline brpc::Channel* get_sparse_channel(size_t server_id) {
        return _server_channels[server_id][0].get();
    }
    inline brpc::Channel* get_dense_channel(size_t server_id) {
        return _server_channels[server_id][1].get();
    }
    inline brpc::Channel* get_cmd_channel(size_t server_id) {
        return _server_channels[server_id][2].get();
    }


    bool _running = false;
    bool _flushing = false;
    std::atomic<uint32_t> _async_call_num; //异步请求计数
    
    std::thread _async_push_dense_thread;
    VectorObjPool _dense_matrix_obj_pool;
    typedef AsyncRequestTask<VectorPooledObj> DenseAsyncTask;
    typedef thread_queue<DenseAsyncTask*, store_value> DenseAsyncTaskQueue;
    std::unordered_map<uint32_t, std::shared_ptr<DenseAsyncTaskQueue>> _push_dense_task_queue_map;

    std::thread _async_push_sparse_thread;
    SparsePushObjPool _sparse_push_obj_pool;
    typedef AsyncRequestTask<SparsePushPooledObj> SparseAsyncTask;
    typedef thread_queue<SparseAsyncTask*, store_value> SparseAsyncTaskQueue;
    std::unordered_map<uint32_t, std::shared_ptr<SparseAsyncTaskQueue>> _push_sparse_task_queue_map;
    std::unordered_map<uint32_t, uint32_t> _push_sparse_merge_count_map;

    int push_sparse_async_shard_merge(std::vector<std::shared_ptr<SparseAsyncTask>>& task_list, 
        std::vector<int>& request_kv_num, int table_id, int shard_idx, ValueAccessor* accessor);
    
    int push_sparse_async_shard_push(std::vector<std::shared_ptr<SparseAsyncTask>>& task_list, 
        std::vector<int>& request_kv_num, int table_id, int shard_idx, DownpourBrpcClosure* closure, ValueAccessor* accessor);

    std::vector<std::shared_ptr<brpc::Channel>> _client_channels;  //client2client 
    std::vector<std::array<std::shared_ptr<brpc::Channel>, 3>> _server_channels; //client2server

    Uint16VecObjPool _dense_compressed_gradient;

private:

    int32_t start_client_service();
    DownpourBrpcClosure* make_push_closure(int request_call_num, PsCmdID cmd);
    void push_dense_raw_gradient(std::shared_ptr<DenseAsyncTask>& task,
            float* total_send_data,
            size_t total_send_data_size,
            DownpourBrpcClosure* closure);
    void push_dense_compress_gradient(std::shared_ptr<DenseAsyncTask>& task,
            float* total_send_data,
            size_t total_send_data_size,
            DownpourBrpcClosure* closure);

    float _mae = 0;
    float _mse = 0;
    uint16_t _push_times = 0;
    brpc::Server _server; 
    DownpourPsClientService _service;
};

}
}
