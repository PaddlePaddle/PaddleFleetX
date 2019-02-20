#include "ps_instance.h"
#include <glog/logging.h>
//#include <mpi.h>
#include "common/archive.h"

namespace paddle {
namespace ps {
    /*
    static MPI_Comm& global_worker_comm() {
        static MPI_Comm worker_comm;
        return worker_comm;
    }

    static MPI_Comm& global_server_comm() {
        static MPI_Comm server_comm;
        return server_comm;
    }

    int MPIPSInstance::init(PSParameter & init_param) {
        //google::InitGoogleLogging("pslib");
        //FLAGS_logtostderr = 1;
        // set ERROR/FATAL to job.err.log
        //FLAGS_stderrthreshold = 2;

        // init mpi environment (startup program by mpi, ssh...)
        LOG(INFO) << "init ps instance by mpi environment";
        _config = init_param;
        int hr = init_env();
        if (hr < 0) {
            LOG(INFO) << "fail to init mpi environment";
        }
        
        return 0;
    }

    int MPIPSInstance::init_env() {
        // todo-low version can't accept NULL
        int hr = MPI_Init(NULL, NULL);
        if (MPI_SUCCESS != hr) {
            LOG(FATAL) << "MPI_init failed with error code" << hr; 
            return -1;
        }    

        //LOG(INFO) << "Succeed to call MPI_Init";
        MPI_Comm_rank(MPI_COMM_WORLD, &_rankid);

        int total_proc = 0;
        MPI_Comm_size(MPI_COMM_WORLD, &total_proc);

        MpiPSInstParameter mpi_param = _config.mpi_psinst_param();
        _server_worker_mode = mpi_param.server_worker_mode(); 
        _proc_per_node = mpi_param.proc_per_node();

        int nodes = mpi_param.nodes();
        _worker_num = nodes;
        _server_num = nodes;
        if (mpi_param.has_worker_num()) {
            _worker_num = mpi_param.worker_num();
        }

        if (mpi_param.has_server_num()) {
            _server_num = mpi_param.server_num();
        }

        if (_rankid == 0) {
            LOG(INFO) << "worker_num: " << _worker_num 
                      << ", server_num: " << _server_num
                      << ", node_num: " << mpi_param.nodes()
                      << ", proc_per_node: " << _proc_per_node
                      << ", server_worker_mode: " << _server_worker_mode
                      << ", total_proc: " << total_proc
                      << ", rankid: " << _rankid;
        }

        _total_server_worker = _worker_num + _server_num;
        if (total_proc != mpi_param.nodes() * _proc_per_node
            || _total_server_worker != _worker_num + _server_num) {
            LOG(FATAL) << "wrong setting: npernode " << _proc_per_node
                << ", worker_num: " << _worker_num
                << ", server_num: " << _server_num;
        } 

        set_nodetype();
         
        // split communicator 
        hr = 0;
        if (is_server()) {
            auto& server_comm = global_server_comm();
            hr = MPI_Comm_split(MPI_COMM_WORLD, 1, _rankid, &server_comm);
        } else if (is_worker()) {
            auto& worker_comm = global_worker_comm();
            hr = MPI_Comm_split(MPI_COMM_WORLD, 0, _rankid, &worker_comm);
        }

        if (MPI_SUCCESS != hr) {
            LOG(FATAL) << "MPI_Comm_split failed";
            return -1;
        }

        return 0;
    }

    void MPIPSInstance::set_nodetype() {
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

    int MPIPSInstance::get_worker_index() {
        if (_server_worker_mode == 0) { 
            return _rankid - _server_num;
        } else {
            return _rankid / _proc_per_node;
        }    
    }

    int MPIPSInstance::get_server_index() {
        if (_server_worker_mode == 0) { 
            return _rankid;
        } else {
            return _rankid / _proc_per_node;
        }    
    }

    bool MPIPSInstance::is_first_worker() {
        return is_worker() && 0 == get_worker_index();
    }


    bool MPIPSInstance::is_first_proc_one_machine() {
        return 0 == _rankid % _proc_per_node;
    }
    
    std::shared_ptr<Worker> MPIPSInstance::get_worker_ptr() {
        if (is_worker()) {
            return _worker;
        } else {
            LOG(ERROR) << "The instance is not a worker, user can not call get_worker_ptr";
            return NULL;
        }
    }
    
    std::shared_ptr<Server> MPIPSInstance::get_server_ptr() {
        if (is_server()) {
            //_server.reset(create_worker(init_param.server_class().c_str()));
            return _server;
        } else {
            LOG(ERROR) << "The instance is not a server, user can not call get_server_ptr";
            return NULL;
        }            
    }

    void MPIPSInstance::barrier_all() {
        MPI_Barrier(MPI_COMM_WORLD);
    }

    // barrier all workers
    void MPIPSInstance::barrier_worker() {
        int hr = MPI_Barrier(global_worker_comm());
        if (MPI_SUCCESS != hr) {
            LOG(ERROR) << "MPI_Barrier worker comm failed";
            return;
        }

        return;
    }

    void MPIPSInstance::worker_all_reduce_sum(double* dst, double* src, int size, bool in_place) {
        if (in_place) {
            MPI_Allreduce(MPI_IN_PLACE, src, size, MPI_DOUBLE, MPI_SUM, global_worker_comm());
        } else {
            MPI_Allreduce(src, dst, size, MPI_DOUBLE, MPI_SUM, global_worker_comm());
        }
    }
    void MPIPSInstance::worker_bcast(BinaryArchive& ar, int len, int root) {
        int send_len = len;
        MPI_Bcast(&send_len, 1, MPI_INT, root, global_worker_comm());

        ar.resize(send_len);
        ar.set_cursor(ar.buffer());
        MPI_Bcast(ar.buffer(), send_len, MPI_BYTE, root, global_worker_comm());
    }

    void MPIPSInstance::finalize() {
        MPI_Finalize();
    }
    */
}
}
