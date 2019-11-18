import sys
import os

class HPCClient(object):
    def __init__(self):
        self.conf_dict = {}

    def print_args(self):
        print("task_name: {}".format(self.task_name))
        print("hdfs_path: {}".format(self.hdfs_path))
        print("ugi: {}".format(self.ugi))
        print("hdfs_output: {}".format(self.hdfs_output))
        print("nodes: {}".format(self.nodes))
        print("hadoop_home: {}".format(self.hadoop_home))
        print("hpc_home: {}".format(self.hpc_home))
        print("train_cmd: {}".format(self.train_cmd))
        print("package_path: {}".format(self.package_path))
        print("priority: {}".format(self.priority))
        print("queue: {}".format(self.queue))
        print("server: {}".format(self.server))
        print("mpi_node_mem: {}".format(self.mpi_node_mem))
        print("pcpu: {}".format(self.pcpu))
        print("python_tar: {}".format(self.python_tar))
        print("wheel: {}".format(self.wheel))

    def check_args(self):
        assert self.task_name != ""
        assert self.hdfs_path != ""
        assert self.ugi != ""
        assert self.hdfs_output != ""
        assert self.nodes != ""
        assert self.hadoop_home != ""
        assert self.hpc_home != ""
        assert self.train_cmd != ""
        assert self.package_path != ""
        assert self.priority != ""
        assert self.queue != ""
        assert self.server != ""
        assert self.mpi_node_mem != ""
        assert self.pcpu != ""
        assert self.python_tar != ""
        assert self.wheel != ""

    def generate_qsub_conf(self, job_dir):
        with open("{}/qsub.conf".format(job_dir), "w") as fout:
            fout.write("SERVER={}\n".format(self.server))
            fout.write("QUEUE={}\n".format(self.queue))
            fout.write("PRIORITY={}\n".format(self.priority))
            fout.write("USE_FLAGS_ADVRES=yes\n")

    def generate_submit_sh(self, job_dir):
        with open("{}/submit.sh".format(job_dir), "w") as fout:
            fout.write("#!/bin/bash\n")
            fout.write("unset http_proxy\n")
            fout.write("unset https_proxy\n")
            fout.write("export HADOOP_HOME={}\n".format(
                self.hadoop_home))
            fout.write("$HADOOP_HOME/bin/hadoop fs -Dhadoop.job.ugi={}"
                       " -Dfs.default.name={} -rmr {}\n".format(
                           self.ugi,
                           self.hdfs_path,
                           self.hdfs_output))
            fout.write("MPI_NODE_MEM={}\n".format(self.mpi_node_mem))
            fout.write("{}/bin/qsub_f -N {} --conf qsub.conf "
                       "--hdfs {} --ugi {} --hout {} --files ./package "
                       "-l nodes={},walltime=1000:00:00,pmem-hard={},"
                       "pcpu-soft={},pnetin-soft=1000,"
                       "pnetout-soft=1000 job.sh\n".format(
                           self.hpc_home,
                           self.task_name,
                           self.hdfs_path,
                           self.ugi,
                           self.hdfs_output,
                           self.nodes,
                           self.mpi_node_mem,
                           self.pcpu))

    def generate_job_sh(self, job_dir):
        with open("{}/job.sh".format(job_dir), "w") as fout:
            fout.write("#!/bin/bash\n")
            fout.write("WORKDIR=`pwd`\n")
            fout.write("mpirun -npernode 1 mv package/* ./\n")
            fout.write("echo 'current dir: '$WORKDIR\n")
            fout.write("mpirun -npernode 1 tar -zxvf python.tar.gz > /dev/null\n")
            fout.write("export LIBRARY_PATH=$WORKDIR/python/lib:$LIBRARY_PATH\n")
            fout.write("mpirun -npernode 1 python/bin/python -m pip install "
                       "{} --index-url=http://pip.baidu.com/pypi/simple "
                       "--trusted-host pip.baidu.com > /dev/null\n".format(
                           self.wheel))
            fout.write("export PATH=python/bin:$PATH\n")
            fout.write("mpirun -npernode 2 -timestamp-output -tag-output -machinefile "
                       "${{PBS_NODEFILE}} python/bin/{}\n".format(self.train_cmd))
            fout.write("if [[ $? -ne 0 ]]; then\n")
            fout.write("    echo 'Failed to run mpi!' 1>&2\n")
            fout.write("    exit 1\n")
            fout.write("fi\n")

    def submit(self, **kwargs):
        # task_name, output_path
        self.task_name = kwargs.get("task_name", "test_submit_job")
        self.hdfs_path = kwargs.get("hdfs_path", "")
        self.ugi = kwargs.get("ugi", "")
        self.hdfs_output = kwargs.get("hdfs_output", "")
        self.nodes = str(kwargs.get("nodes", 1))
        self.hadoop_home = kwargs.get("hadoop_home", "")
        self.hpc_home = kwargs.get("hpc_home", "")
        self.train_cmd = kwargs.get("train_cmd", "")
        self.package_path = kwargs.get("package_path", "")
        self.priority = kwargs.get("priority", "")
        self.queue = kwargs.get("queue", "")
        self.server = kwargs.get("server", "")
        self.mpi_node_mem = str(kwargs.get("mpi_node_mem", 11000))
        self.pcpu = str(kwargs.get("pcpu", 180))
        self.python_tar = kwargs.get("python_tar", "")
        self.wheel = kwargs.get("wheel", "")

        self.print_args()
        self.check_args()
        jobdir = "{}_jobdir".format(self.task_name)
        os.system("mkdir -p {}_jobdir".format(self.task_name))
        os.system("rm -rf {}/package".format(jobdir))
        os.system("cp -r {} {}/package".format(self.package_path, jobdir))
        os.system("cp {} {}/package/".format(self.python_tar, jobdir))
        os.system("cp {} {}/package/".format(self.wheel, jobdir))
        # make submit dir
        self.generate_submit_sh(jobdir)
        # generate submit.sh
        self.generate_job_sh(jobdir)
        # generate job.sh
        self.generate_qsub_conf(jobdir)
        # run submit
        os.system("cd {};sh submit.sh".format(jobdir))
