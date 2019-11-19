from paddle.fluid.incubate.fleet.utils.hdfs import HDFSClient

def hdfs_ls(path, fs_name, ugi):
    configs = {
        "fs.default.name": fs_name,
        "hadoop.job.ugi": ugi,
    }
    hdfs_client = HDFSClient("$HADOOP_HOME", configs)
    filelist = []
    for i in path:
        cur_path = hdfs_client.ls(i)
        if fs_name.startswith("hdfs:"):
            cur_path = ["hdfs:" + j for j in cur_path]
        elif fs_name.startswith("afs:"):
            cur_path = ["afs:" + j for j in cur_path]
        filelist += cur_path
    return filelist
