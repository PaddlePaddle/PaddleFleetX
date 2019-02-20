#include "ps_instance.h"
#include <glog/logging.h>
#include "common/archive.h"

namespace paddle {
namespace ps {

    int PaddlePSInstance::init(PSParameter & init_param) {
        //google::InitGoogleLogging("pslib");
        //FLAGS_logtostderr = 1;
        // set ERROR/FATAL to job.err.log
        //FLAGS_stderrthreshold = 2;
        LOG(ERROR) << "[error]dont use this, please use init(PSParameter & init_param, int rank_id, int server_worker_mode, int proc_per_node, int nodes)";
        return 0;
    }
    
    int PaddlePSInstance::init(PSParameter & init_param, int rank_id, int server_worker_mode, int proc_per_node, int nodes) {
         LOG(INFO) << "init ps instance by paddle environment";

        _config = init_param;
        _rankid = rank_id;
        _server_worker_mode = server_worker_mode; 
        _proc_per_node = proc_per_node;

        _worker_num = nodes;
        _server_num = nodes;

        if (_rankid == 0) {
            LOG(INFO) << "worker_num: " << _worker_num 
                      << ", server_num: " << _server_num
                      << ", node_num: " << nodes
                      << ", proc_per_node: " << _proc_per_node
                      << ", server_worker_mode: " << _server_worker_mode
                      << ", rankid: " << _rankid;
        }

        _total_server_worker = _worker_num + _server_num;
        if ( _total_server_worker != _worker_num + _server_num) {
            LOG(FATAL) << "wrong setting: npernode " << _proc_per_node
                << ", worker_num: " << _worker_num
                << ", server_num: " << _server_num;
        } 

        set_nodetype();
        return 0;   
    }
    int PaddlePSInstance::init_env() {
        return 0;
    }

    void PaddlePSInstance::set_nodetype() {
        if (_server_worker_mode == 0) {
            if (_rankid < _server_num) {
                _node_type = NODE_TYPE_SERVER;
            } else if (_rankid < _total_server_worker) {
                _node_type = NODE_TYPE_WORKER;
            } else {
                _node_type = NODE_TYPE_IDLE;
            }
        } else  if (_server_worker_mode == 1) {
            if (_rankid < _total_server_worker) {
                if (0 == _rankid % _proc_per_node) {
                    _node_type = NODE_TYPE_SERVER;
                } else {
                    _node_type = NODE_TYPE_WORKER;
                }
            } else {
                _node_type = NODE_TYPE_IDLE;
            }
        }

        if (_rankid == 0) {
            LOG(INFO) << "node type: " << _node_type;
        }
    }

    int PaddlePSInstance::get_worker_index() {
        if (_server_worker_mode == 0) { 
            return _rankid - _server_num;
        } else {
            return _rankid / _proc_per_node;
        }    
    }

    int PaddlePSInstance::get_server_index() {
        if (_server_worker_mode == 0) { 
            return _rankid;
        } else {
            return _rankid / _proc_per_node;
        }    
    }

    bool PaddlePSInstance::is_first_worker() {
        return is_worker() && 0 == get_worker_index();
    }


    bool PaddlePSInstance::is_first_proc_one_machine() {
        return 0 == _rankid % _proc_per_node;
    }
    
    void PaddlePSInstance::barrier_all() {
        LOG(ERROR) << "[error] don't use barrier !!!!!";
    }

    void PaddlePSInstance::barrier_worker() {
        LOG(ERROR) << "[error] don't use barrier !!!!!";
    }

    void PaddlePSInstance::finalize() {
        LOG(ERROR) << "[error] don't use finalize !!!!!";
    }
}
}
