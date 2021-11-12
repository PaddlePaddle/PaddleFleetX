import time

# 开始时间
start_time = None

def start_timer():
   # 获取开始时间
   global start_time
   start_time = time.time()

def end_timer_and_print(msg):
   # 打印信息并输出训练时间
   end_time = time.time()
   print("\n" + msg)
   print("共计耗时 = {:.3f} sec".format(end_time - start_time))

import paddle
import paddle.nn as nn

class SimpleNet(nn.Layer):

   def __init__(self, input_size, output_size):
      super(SimpleNet, self).__init__()
      self.linear1 = nn.Linear(input_size, output_size)
      self.relu1 = nn.ReLU()
      self.linear2 = nn.Linear(input_size, output_size)
      self.relu2 = nn.ReLU()
      self.linear3 = nn.Linear(input_size, output_size)

   def forward(self, x):

      x = self.linear1(x)
      x = self.relu1(x)
      x = self.linear2(x)
      x = self.relu2(x)
      x = self.linear3(x)

      return x

epochs = 5
input_size = 4096   # 设为较大的值
output_size = 4096  # 设为较大的值
batch_size = 512    # batch_size 为8的倍数
nums_batch = 50

train_data = [paddle.randn((batch_size, input_size)) for _ in range(nums_batch)]
labels = [paddle.randn((batch_size, output_size)) for _ in range(nums_batch)]

mse = paddle.nn.MSELoss()

model = SimpleNet(input_size, output_size)  # 定义模型

optimizer = paddle.optimizer.SGD(learning_rate=0.0001, parameters=model.parameters())  # 定义优化器

start_timer() # 获取训练开始时间

for epoch in range(epochs):
   datas = zip(train_data, labels)
   for i, (data, label) in enumerate(datas):

      output = model(data)
      loss = mse(output, label)

      # 反向传播
      loss.backward()

      # 训练模型
      optimizer.step()
      optimizer.clear_grad()

print(loss)
end_timer_and_print("默认耗时:") # 获取结束时间并打印相关信息

model = SimpleNet(input_size, output_size)  # 定义模型

optimizer = paddle.optimizer.SGD(learning_rate=0.0001, parameters=model.parameters())  # 定义优化器

# Step1：定义 GradScaler，用于缩放loss比例，避免浮点数溢出
scaler = paddle.amp.GradScaler(init_loss_scaling=1024)

start_timer() # 获取训练开始时间

for epoch in range(epochs):
   datas = zip(train_data, labels)
   for i, (data, label) in enumerate(datas):

      # Step2：创建AMP上下文环境，开启自动混合精度训练
      with paddle.amp.auto_cast():
            output = model(data)
            loss = mse(output, label)

      # Step3：使用 Step1中定义的 GradScaler 完成 loss 的缩放，用缩放后的 loss 进行反向传播
      scaled = scaler.scale(loss)
      scaled.backward()

      # 训练模型
      scaler.minimize(optimizer, scaled)
      optimizer.clear_grad()

print(loss)
end_timer_and_print("使用AMP模式耗时:")