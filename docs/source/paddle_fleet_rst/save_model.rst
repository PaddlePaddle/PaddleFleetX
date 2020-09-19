背景
~~~~

FleetX的模型及其丰富，但即使如此，也常遇到用户想要个性化的模型，但是FleetX暂时不支持的情况。为了帮助用户使用自己组网的模型，本篇文章介绍如何在FleetX中添加新的模型。

我们都知道，在Paddle网络的训练过程中，需要定义以下几个组件：

-  reader
-  前向网络
-  优化器

在保存FleetX时，用户只需要准备reader和前向网络即可。下面以Bert-Large为例，带大家走完保存、运行模型的流程。

step1 安装FleetX
~~~~~~~~~~~~~~~~

step2 保存前向网络
~~~~~~~~~~~~~~~~~~

先下载原始网络代码：

::

   git clone https://github.com/PaddlePaddle/models.git
   cd models/PaddleNLP/pretrain_language_models/BERT

在\ ``train.py``\ 中将定义模型的代码替换为：

.. code:: python

   import fleetx as X
   generator = fluid.unique_name.UniqueNameGenerator()
   with fluid.program_guard(train_program, startup_prog):
       with fluid.unique_name.guard(generator):
           train_data_loader, next_sent_acc, mask_lm_loss, total_loss = create_model(bert_config=bert_config)
           X.util.save_program(
               main_prog=train_program,
               startup_prog=startup_prog,
               program_path="Bert-Large",
               input_list=['src_ids', 'pos_ids', 'sent_ids', 'input_mask', 'mask_label', 'mask_pos', 'labels'],
               hidden_vars=None,
               loss=total_loss,
               generator_info=generator.ids,
               target=[next_sent_acc, mask_lm_loss],
               checkpoints=None,
               learning_rate=None)

其中： - ``main_prog``\ 和\ ``startup_prog``\ 是Paddle的网络； -
``program_path``\ 是你要保存的文件夹名称； -
``input_list``\ 是你的网络的输入； - ``hidden_vars`` -
``loss``\ 是要优化的loss； - ``generator_info``\ 是生成Var
name的generator的id； - ``target``\ 表示要fetch的Variables; -
``checkpoints``\ 是Recompute要用的，不需要的话，可以先设置为None; -
``learning_rate``

需要注意的是： -
模型保存时应该仅仅保存网络结构，不要保存Reader相关的Op，如果你的网络中使用了DataLoader，应该设置iterable=True的选项。

step2 在FleetX添加一个application
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

在FleetX的源码中，在\ ``FleetX/python/fleetx/applications/``\ 路径下，在model.py中添加一个Model的子类：

::

   class BertLarge(ModelBase):
       def __init__(self):
           super(BertLarge, self).__init__()
           fleet_path = sysconfig.get_paths()["purelib"] + '/fleetx/applications/'
           model_name = 'bert_large'
           download_model(fleet_path, model_name)
           inputs, loss, startup, main, unique_generator, checkpoints, target = load_program(
               fleet_path + model_name)
           self.startup_prog = startup
           self.main_prog = main
           self.inputs = inputs
           self.loss = loss
           self.checkpoints = checkpoints
           self.target = target

       def load_digital_dataset_from_file(self,
                                          data_dir,
                                          vocab_path,
                                          batch_size=16,
                                          max_seq_len=128,
                                          in_tokens=False):
           return load_bert_dataset(
               data_dir,
               vocab_path,
               inputs=self.inputs,
               batch_size=batch_size,
               max_seq_len=max_seq_len,
               in_tokens=in_tokens)

在定义子类时，在\ ``__init__()``\ 函数中初始化你的一些变量；在\ ``load_digital_dataset_from_file()``\ 中定义你的reader。

Step4 开始运行
~~~~~~~~~~~~~~

::

   cd FleetX/example
   fleetrun bert_app.py
