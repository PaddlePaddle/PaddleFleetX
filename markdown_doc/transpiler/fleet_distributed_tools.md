# 分布式周边工具使用文档
分布式工具使用文档，主要覆盖模型解析、参数验证、debug模式参数打印等工具。本文档会长期更新～

## 工具一：比较train_program和pruned_program
### 接口：
- check_two_programs(config)

### 功能说明：
- 检查pruned program中的persistable vars的name和shape与裁剪前的train program是否一致，若一致，返回True，反之则返回False

### 用法示例：
```python
from paddle.fluid.incubate.fleet.utils.fleet_util import FleetUtil
 
 
class config:
    pass
conf = config()
conf.train_prog_path = "train_program.pbtxt" # train_program模型文件路径
conf.is_text_train_program = True # train_program格式，二进制设为False，可读的文本格式设为True
 
conf.pruned_prog_path = "pruned_program.pbtxt" # pruned_program模型文件路径
conf.is_text_pruned_program = True # pruned_program格式，二进制设为False，可读的文本格式设为True
conf.draw = True # 是否需要绘制pruned_program的拓扑图
conf.draw_out_name = "pruned_check" # pruned_program拓扑图文件名，拓扑图文件保存在和pruned_program.pbtxt同一目录下
fleet_util = FleetUtil()
res = fleet_util.check_two_programs(conf)
```

## 工具二：模型保存参数验证
### 接口：
- check_vars_and_dump(config)

### 功能说明：
- 首先根据模型路径配置，加载模型和保存的模型参数，若保存的参数在program中不存在或shape不一致，则报错提示。然后根据提供的feed_vars的信息，运行program得到fetch_vars的结果。

### 用法示例：
```python
from paddle.fluid.incubate.fleet.utils.fleet_util import FleetUtil
 
# feed vars info
feed_config = config()
feed_config.feeded_vars_names = ['concat_1.tmp_0', 'concat_2.tmp_0']
feed_config.feeded_vars_dims = [682, 1199]
feed_config.feeded_vars_types = [np.float32, np.float32]
feed_config.feeded_vars_filelist = ["./concat_1", "./concat_2"]   # feed数据来源，len(feeded_vars_filelist)=len(feeded_vars_names). 如果该参数为None，则随机生成若干数作为feed_vars输入
 
fetch_config = config()
fetch_config.fetch_vars_names = ['similarity_norm.tmp_0']  # 如设置为None，则默认使用program中的fetch_target
 
conf = config()
conf.batch_size = 1
conf.feed_config = feed_config      # feed vars配置
conf.fetch_config = fetch_config    # fetch vars配置
conf.dump_model_dir = "./"  # program和模型参数存储地址
conf.dump_program_filename = "pruned_program.pbtxt"    # program文件名称，在conf.dump_model_dir中。
conf.is_text_dump_program = True  # program是否是human-readable text形式；否则是binary string形式
# 如果所有参数存在一个文件中，需要指定文件名，从conf.dump_model_dir/conf.save_params_filename读取；如果参数存在多个文件中，设置为None，从conf.dump_model_dir读取
conf.save_params_filename = None
 
fleet_util = FleetUtil()
results = fleet_util.check_vars_and_dump(conf)
```

## 工具三：模型拓扑图绘制
### 接口
- draw_from_program_file(model_filename, is_text, output_dir, output_filename)
- draw_from_program(program, output_dir, output_filename)

### 功能说明：
- 利用graphviz绘制模型拓扑图，格式为pdf文件，用以检查裁剪后的program是否符合预期。**如program过大，使用pdf阅读器无法完全打开，可尝试在浏览器中打开查看。**

### 用法示例：
```python
from paddle.fluid.incubate.fleet.utils.fleet_util import FleetUtil
import paddle.fluid.incubate.fleet.utils.utils as utils
 
program_path = "train_program.pbtxt" # program文件名称
is_text = True # program是否是human-readable text形式；否则是binary string形式
output_dir = "./" # 输出目录
output_filename = "draw_program" # 拓扑图文件名，保存在output_dir目录下
 
 
fleet_util = FleetUtil()
fleet_util.draw_from_program_file(program_path, is_text, output_dir, output_filename)
 
 
program = utils.load_program(program_path, is_text)
fleet_util.draw_from_program(program, output_dir, output_filename)
```

## 工具四：模型proto文件解析
### 接口：
- parse_program_proto(prog_path, is_text, output_dir)

### 功能说明：
- 解析program proto中的persistable vars， all vars以及ops，方便排查。该工具产出为三个文件，output_dir/vars_all.log；output_dir/vars_persistable.log； output_dir/ops.log。

### 用法示例：
```python
from paddle.fluid.incubate.fleet.utils.fleet_util import FleetUtil
 
 
prog_path = "./pruned_program.pbtxt" # program文件名称
is_text = True # program是否是human-readable text形式；否则是binary string形式
output_dir = "./" # 输出目录
 
 
fleet_util = FleetUtil()
fleet_util.parse_program_proto(prog_path, is_text, output_dir)
```

## 工具五：格式转换
### 接口：
- program_type_trans(prog_dir, prog_fn, is_text)

### 功能说明：
- program二进制格式和文本格式转换。

### 用法示例：
```python
from paddle.fluid.incubate.fleet.utils.fleet_util import FleetUtil
 
prog_dir = "./"
prog_fn = "pruned_program.bin"
is_text = False
 
 
fleet_util = FleetUtil()
fleet_util.program_type_trans(prog_dir, prog_fn, is_text)
```

## 工具六：Debug模式Dump参数/梯度
*注：该工具目前只在dataset-debug模式下支持*

### 使用方法：
#### 1. 配置需要dump的参数和梯度信息
- **字段定义：**
    - dump_field: 需要dump的op输出的var name，每条样本dump一次
    - dump_param: 需要dump的模型参数和梯度，每个batch dump一次
    - dump_fields_path: dump结果文件保存路径，支持hdfs路径和本地路径，若该路径为hdfs路径，需以afs:或hdfs:开头，且在dataset定义中利用set_hdfs_config函数设置hadoop配置
    
- **transpiler模式配置方法**（transpiler模式目前只在distributedStrategy中支持）：
```python
# transpiler模式目前只在distributedStrategy中支持
 
 
from paddle.fluid.incubate.fleet.parameter_server.distribute_transpiler import fleet
from paddle.fluid.incubate.fleet.parameter_server.distribute_transpiler.distributed_strategy import StrategyFactory
 
loss = net() # 网络定义
strategy = StrategyFactory.create_async_strategy()
strategy.set_debug_opt({
    "dump_param": ["fc_0.w_0"],
    "dump_fields": ["fc_0.tmp_0"],
    "dump_fields_path": "afs:/user/dump_text"
})
adam = fluid.optimizer.Adam(learning_rate=0.000005)
adam = fleet.distributed_optimizer(adam, strategy)
adam.minimize(loss)
```
- **pslib模式配置方法**
```python
from paddle.fluid.incubate.fleet.parameter_server.pslib import fleet
loss = net() # 网络定义
adam = fluid.optimizer.Adam(learning_rate=0.000005)
adam = fleet.distributed_optimizer(adam, strategy={"use_cvm" : True, "fleet_desc_file" : "cupai_fleet_desc.prototxt", "dump_fields": ["fc_0.tmp_0"], "dump_fields_path": "afs:/user/dump_text/"})
adam.minimize(loss)
```
#### 2. 返回结果：
- dump_filed字段中的参数，返回格式如下：
```python
ins_id \t ins_content \t field1:length:0.0:0.1:...:0.2 \t field2:length:0.0:0.1:...:0.2
```
*ins_id, ins_content默认为空，但因为dump参数的过程是多线程并发的，所以输出结果和训练集ins顺序并非一一对应，为方便debug，推荐设置ins_id。*

- dump_param字段中参数，返回格式如下：
```python
(batch_id, param_name):0.1:0.2:...:0.3
```
- 示例
```python
16a9f1775e0001017ed560157aa07fbf\t\tdnn-fc-3.tmp_0:1:0.0231031\tdnn-fc-3.tmp_0@GRAD:1:0.00081847
(2,concat_0.tmp_0):0.0209947:0.0381233:0.0209938:0.0336022:0.0209938:0.0381233:0.0209945:0.0410865:0.0209955:0.0379817:0.0209938:0.0336022:0.0209945:0.0412028:0.0209938:0.0336022:0.0209938:0.0336022:0.0209938:0.0336022:0.0209945:0.0482546:0.020994:0.0381233:0.0209946:0.0380605:0.0209938:0.0335739:0.0209938:0.0336022:0.0209941:0.037924:0.0209947:0.0336022:0.020994:0.0379511:0.0209938:0.0336022:0.020994:0.0369887:0.0209938:0.0335739:0.0209946:0.0381233:0.0209938:0.0336022:0.0209938:0.0336022:0.0209938:0.0375006:0.020994:0.0369887:0.0209946:0.0377272:0.0209946:0.0379539:0.0209938:0.0336022:0.020994:0.0369887:0.0209946:0.0380086:0.0209945:0.0412018:0.0209938:0.0336022:0.0209938:0.0336022:0.0209945:0.03801:0.0209947:0.0410887:0.020994:0.0379229:0.0209938:0.0336022:0.020996:0.0444872:0.0209938:0.0336022:0.0209938:0.0336022:0.0209941:0.0381233:0.0209938:0.0375006:0.0209972:0.0483415:0.0209946:0.0381233:0.0209986:0.0483708:0.0209938:0.0336022:0.0209938:0.0336022:0.0209938:0.0381233:0.020994:0.0381233:0.0209938:0.0336022:0.0209938:0.0380103:0.0209955:0.0380667:0.0209938:0.0336022:0.020994:0.0369887:0.0209945:0.0412028:0.0209946:0.0381233:0.0209972:0.0483708:0.0209945:0.0409753:0.0209938:0.0336022:0.0209956:0.0483143:0.0209972:0.0483708:0.0209946:0.0380605:0.0209938:0.0336022:0.0209938:0.0336022:0.0209946:0.0379539:0.020994:0.0369887:0.0209945:0.0412028:0.0209938:0.0336022:0.0209938:0.0336022:0.0209938:0.0336022:0.0209947:0.0336022:0.0209938:0.0336022:0.0209938:0.0336022:0.0209957:0.0412028:0.0209938:0.0336022:0.0209946:0.0380605:0.0209938:0.0378688:0.0209947:0.0336022:0.020994:0.0369887:0.020994:0.0369887:0.020994:0.0369887:0.0209938:0.0336022:0.020994:0.0369887:0.0209957:0.0482858:0.0209938:0.0336022:0.0209938:0.0381233:0.0209938:0.0380669:0.0209938:0.0336022:0.0209941:0.0378967:0.0209938:0.0381233:0.0209938:0.0336022:0.0209945:0.0411181:0.0209938:0.0336022:0.0209945:0.0409753:0.0209938:0.0336022:0.020994:0.0369887:0.0209947:0.0336022:0.0209947:0.0410894:0.0209938:0.0336022:0.020994:0.0369887:0.020994:0.0380667:0.0209938:0.0381233:0.020996:0.0444601:0.0209938:0.0336022:0.0209945:0.0409753:0.0209938:0.0336022:0.0209938:0.0336022:0.0209945:0.0412573:0.0209945:0.0410865:0.0209938:0.0381233:0.0209938:0.0336022:0.0209938:0.0336022:0.0209945:0.03801:0.0209938:0.0336022:0.0209984:0.0483425:0.020994:0.0381233:0.020994:0.0369887:0.0209947:0.0410887:0.0209945:0.0482546:0.0209938:0.0336022:0.0209949:0.0379783:0.0209956:0.0483708:0.0209945:0.03801:0.0209938:0.0336022:0.0209938:0.0336022:0.0209946:0.0377544:0.0209947:0.0483708
```
#### 3. 设置ins_id, ins_content方法：
1. 在dataset_generator中定义ins_id, ins_content字段内容
2. 在dataset中解析ins_id, ins_content，具体函数接口为dataset.set_parse_ins_id(True), dataset.set_parse_content(True)
3. 示例
```python
# dataset_generator.py相关代码
class MyDataset(dg.MultiSlotStringDataGenerator):
    def load_resource(self, dictf):
        self._all_slots_dict = collections.OrderedDict()
        with open(dictf, 'r') as f:
            slots = f.readlines()
        for index, slot in enumerate(slots):
            self._all_slots_dict[slot.strip()] = [False, index + 4]
  
    def generate_sample(self, line):
        def data_iter():
            elements = line.split('\t')[0].split()[1:]
            ins_id = line.split('\t')[0].split()[0]
            padding = 0
            ## !!! 增加ins_id, ins_content字段
            output = [("insid", [ins_id]), ("ins_content", elements), ("show", [elements[0]]), ("click", [elements[1]])]
            output += [(slot, []) for slot in self._all_slots_dict]
            for elem in elements[2:]:
                feasign, slot = elem.split(':')
                if slot == "12345":
                    feasign = feasign
                else:
                    feasign = feasign
                if not self._all_slots_dict.has_key(slot):
                    continue
                self._all_slots_dict[slot][0] = True
                index = self._all_slots_dict[slot][1]
                output[index][1].append(feasign)
            for slot in self._all_slots_dict:
                visit, index = self._all_slots_dict[slot]
                if visit:
                    self._all_slots_dict[slot][0] = False
                else:
                    output[index][1].append(str(padding))
            yield output
        return data_iter
 
 
# dataset定义中解析ins_id, ins_content
dataset = fluid.DatasetFactory().create_dataset(e)
dataset.set_use_var([])
dataset.set_batch_size(32)
dataset.set_thread(12)
dataset.set_hdfs_config()
dataset.set_pipe_command()
dataset.set_parse_ins_id(True)
dataset.set_parse_content(True)
```