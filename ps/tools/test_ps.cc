#include "ps_instance.h"
#include "communicate/ps_client.h"
#include "communicate/ps_server.h"
#include "common/afs_warpper.h"
#include "common/fs.h"
#include "proto/ps.pb.h"


int main(int argc, char *argv[])
{
    
    #ifdef MPI_TRAIN_ENV
        paddle::ps::LegoAsyncInstance * ps_inst = new paddle::ps::LegoAsyncInstance();
        paddle::ps::LegoAsyncWorker * worker = new paddle::ps::LegoAsyncWorker();
        delete ps_inst;
        delete worker;
    #endif

    {
    
        ::paddle::ps::AfsClient afs_client;
        ::paddle::FsClientParameter afs_cfg;
        std::string hdfs_command = "/home/work/afs_client2/hadoop-client/hadoop/bin/hadoop fs -Dfs.default.name=afs://xingtian.afs.baidu.com:9902 -Dhadoop.job.ugi=mlarch,Fv1M87 -Ddfs.client.block.write.retries=15 -Ddfs.rpc.timeout=300000";
        afs_cfg.set_hadoop_bin(hdfs_command);
        afs_cfg.set_buffer_size(0);
        afs_client.initialize(afs_cfg);
        std::vector<std::string> fs = afs_client.list("afs:/user/mlarch_rd/renke");
        if (fs.size() > 0) {
            std::cout << "test ls : " << fs[0] << std::endl;
        } else {
            std::cout << "faile" << std::endl;
        }
        ::paddle::ps::fs_mkdir("afs://xingtian.afs.baidu.com:9902//user/mlarch_rd/renke/test");
        if (afs_client.exist("afs:/user/mlarch_rd/renke/test")) {
            std::cout << "path exists" << std::endl;
        }
        afs_client.remove_dir("afs:/user/mlarch_rd/renke/test");
        if (afs_client.exist("afs:/user/mlarch_rd/renke/test")) {
            std::cout << "path exists" << std::endl;
        } else {
            std::cout << "path not exists" << std::endl;
        }
    
    }
    return 0;
}

