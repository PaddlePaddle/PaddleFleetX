## 使用fleetsub提交集群任务

### `fleetsub`是什么

当您安装了`fleet-x`后，便可以使用`fleetsub`在集群上提交分布式任务。长期的目标是成为集群任务提交的统一命令，只需要一行启动命令，就会将训练任务提交到离线训练集群中。目前该功能只支持百度公司内部云上的任务提交，使用`fleetsub`前需要先安装paddlecloud客户端，后续我们会支持更多的公有云任务提交。

### 使用要求
使用`fleetsub`命令的要求：安装`fleet-x`

- 【方法一】从pip源安装

``` sh
    pip install fleet-x
```

- 【方法二】下载whl包并在本地安装

``` sh
    # python2
    wget --no-check-certificate https://fleet.bj.bcebos.com/fleet_x-0.0.4-py2-none-any.whl
    pip install fleet_x-0.0.4-py2-none-any.whl
    # python3
    wget --no-check-certificate https://fleet.bj.bcebos.com/fleet_x-0.0.4-py3-none-any.whl
    pip3 install fleet_x-0.0.4-py3-none-any.whl
```

### 使用说明

在提交任务前，用户需要在yaml文件中配置任务相关的信息，如：节点数、镜像地址、集群信息、启动训练所需要的命令等。

首先看一个yaml文件的样例。因为信息安全的原因yaml文件中的信息做了脱敏。

``` yaml
num_trainers: 4
num_cards: 8
job_prefix: bert_base_pretraining
image_addr: ${image_addr:-"dockhub.com/paddlepaddle-public/paddle_ubuntu1604:cuda10.0-cudnn7-dev"}
cluster_name: v100-32-cluster
group_name: k8s-gpu-v100-8
server: paddlecloud.server.com
log_fs_name: "afs://xx.fs.com:9902"
log_fs_ugi: "ugi_name,ugi_passwd"
log_output_path: "/xx/yy/zz"
file_dir: "./"

whl_install_commands:
  - pip install fleet_x-0.0.5-py2-none-any.whl
  - pip install paddlepaddle_gpu-0.0.0-cp27-cp27mu-linux_x86_64.whl

commands:
  - fleetrun bert_base.py --download_config=bert.yaml

```

| 字段名称 | 字段含义 | 类型 |
|   ----   |   ----   | ---- |
| num_trainers | 训练节点的数量 | INT |
| num_cards    | 单节点上申请的GPU卡数 | INT |
| job_prefix   | 任务名前缀 | STRING |
| image_addr   | 镜像地址   | {STRING} |
| cluster_name | 集群名     | STRING |
| group_name   | 群组名     | STRING |
| server       | 集群master节点服务名 | STRING |
| log_fs_name  | 任务日志存放的文件系统名 | STRING |
| log_fs_ugi   | 任务日志存放的文件系统UGI | STRING |
| log_output_path | 任务日志存放的目标文件系统地址 | STRING |
| file_dir     | 提交任务需要上传的文件目录 | STRING |
| whl_install_commands | 安装各种wheel包的命令 | Repeated Command Line |
| commands | 运行任务执行的各种命令 | Repeated Command Line |

### 任务提交

定义完上述脚本后，用户即可使用`fleetsub`命令向PaddleCloud 提交任务了：

``` sh
    fleetsub -f demo.yaml
```

### 使用样例

具体的使用说明及样例代码请参考下面的[WIKI](http://wiki.baidu.com/pages/viewpage.action?pageId=1236728968)

    

