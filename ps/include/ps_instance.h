#pragma once
#include <memory>
//#include "node.h"
#include "proto/ps.pb.h"
#include "ps_global_info.h"
#include "common/archive.h"

namespace paddle {
namespace ps {

/**
 * PSInstance holds global information about current job
 * A new kind of PSInstance should inherit PSInstance and implement needed interfaces
 **/
class PSInstance {
public:
    static PSInstance* instance() { return NULL; }

    explicit PSInstance() {}
    virtual ~PSInstance() {}

    /**
     * @brief init(PSParameter & init_param) is responsible for initialization
     *        of the current instance, 
     * @param [in] init_param: PSParameter & init_param, 
     *        PSParmeter is defined in a protobuf
     * @return int: = 0 success; otherwise fail
    **/
    virtual int init(PSParameter & init_param) = 0;
    virtual int init(PSParameter & init_param, int rank_id, int server_worker_mode, int proc_per_node, int nodes) = 0;

    /**
     * @brief when a job fails or stops, finalize will be called
     **/
    virtual void finalize() = 0;

    /**
     * @brief barrier_all will hang all nodes
     * @return void
     **/
    virtual void barrier_all() = 0;

    /**
     * @brief barrier_worker will hang all worker nodes
     * @return void
     **/
    virtual void barrier_worker() = 0;

    /**
     * @brief if current instance is a server
     * @return true if server, false if worker
     **/
    virtual bool is_server() { return false; };

    /**
     * @brief if current instance is a server
     * @return true if server, false if worker
     **/
    virtual bool is_worker() { return false; };

    virtual PSParameter& get_param() { return _param; }
protected:
    PSParameter _param;
};


class PaddlePSInstance : public PSInstance {
public:
    explicit PaddlePSInstance() {}
    virtual ~PaddlePSInstance() {}

    /**
     * @brief initance() is responsible to get psinstance object
     * @return PaddlePSInstance* - psinstance ptr
    **/
    static PaddlePSInstance* instance() {
        static PaddlePSInstance inst;
        return &inst;
    }

    /**
     * @brief init(PSParameter & init_param) is responsible for initialization
     *        of the current instance, 
     * @param [in] init_param: PSParameter & init_param, 
     *        PSParmeter is defined in a protobuf
     * @return void
    **/
    virtual int init(PSParameter & init_param);
    virtual int init(PSParameter & init_param, int rank_id, int server_worker_mode, int proc_per_node, int nodes);

    /**
     * @brief get_ps_parameter() will return ps parameter
     * @return PSParameter: ps parameter 
    **/
    virtual PSParameter& get_config() { return _config; }

    /**
     * @brief when a job fails or stops, finalize will be called
     **/
    virtual void finalize();

    /**
     * @brief barrier_all will hang all nodes
     * @return void
     **/
    virtual void barrier_all();

    /**
     * @brief barrier_worker will hang all worker nodes
     * @return void
     **/
    virtual void barrier_worker();

    /**
     * @brief if current instance is a server
     * @return true if server, false if worker
     **/
    virtual bool is_server() { return _node_type == NODE_TYPE_SERVER; }

    /**
     * @brief if current instance is a server
     * @return true if server, false if worker
     **/
    virtual bool is_worker() { return _node_type == NODE_TYPE_WORKER; }

     /**
     * @brief if current instance is the first worker, 
     * @return bool - true: is first worker, otherwise false
     **/
    virtual bool is_first_worker();

     /**
     * @brief get total worker num 
     * @return int - worker count
     **/
    virtual int get_worker_num() { return _worker_num; }

     /**
     * @brief get the current worker's index in total workers
     * @return int - worker index, start from 0 to _worker_num-1
     **/
    virtual int get_worker_index();

     /**
     * @brief get the current server's index in total servers
     * @return int - server index, start from 0 to _server_num-1
     **/
    virtual int get_server_index();

     /**
     * @brief get if the current proc is the first proc in one machine
     * @return bool - true: is the first proc in the current machine, otherwise false
     **/
    virtual bool is_first_proc_one_machine();
   
    /**
     * @brief get the current process's id
     * @return int - process id,  from 0 to total process 
     **/
    virtual int get_rankid() { return _rankid; }

public:
    enum node_type_t {
        NODE_TYPE_SERVER = 0,
        NODE_TYPE_WORKER = 1,
        NODE_TYPE_IDLE = 2
    };

protected:
    /**
     * @brief init_env() is responsible for initialization
     *        of the current instance if the program is start up by mpirun, 
     * @return void
    **/
    virtual int init_env();

    /**
     * @brief if current instance is a server, 
     *        set_server_ptr() will set server pointer
     * @param [in] ptr: server node ptr 
     * @return void 
     **/
    virtual void set_nodetype();
    
protected:

    // process id from mpi, start from 0 
    int _rankid;

    // process num in one machine (default: 2)
    int _proc_per_node;

    // total process num from mpi
    int _total_server_worker;
    
    // server num
    int _server_num;

    // worker num
    int _worker_num;

    /*
    // server MPI COMM
    MPI_Comm _server_comm;

    // worker MPI COMM
    MPI_Comm _worker_comm;
    */

    // node type: server, worker, others
    node_type_t _node_type;

    // mode 0: server0, server1,...,worker0, worker1
    // mode 1(default): server0, worker0, server1, worker1...
    int _server_worker_mode;

    // parameter for ps initialization
    PSParameter _config;
};
}
}
