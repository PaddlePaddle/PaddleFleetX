#ifndef BAIDU_BAIDU_PSLIB_INCLUDE_PS_GLOBAL_INFO_H
#define BAIDU_BAIDU_PSLIB_INCLUDE_PS_GLOBAL_INFO_H

namespace paddle {
namespace ps {
class PSGlobalInfo {
public:
    explicit PSGlobalInfo() : 
        _worker_num(0), 
        _server_num(0), 
        _shard_num(-1) {}
    virtual ~PSGlobalInfo() {}
    void set_worker_num(int num) { _worker_num = num; }
    void set_server_num(int num) { _server_num = num; }
    void set_shard_num(int num) { _shard_num = num; }
private:

    int _worker_num;
    int _server_num;
    int _shard_num;
};
}
}
#endif
