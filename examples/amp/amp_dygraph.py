import time

start_time = None

def start_timer():
   global start_time
   start_time = time.time()

def end_timer_and_print(msg):
   end_time = time.time()
   print("\n" + msg)
   print("total time cost = {:.3f} sec".format(end_time - start_time))

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
input_size = 4096
output_size = 4096
batch_size = 512
nums_batch = 50

train_data = [paddle.randn((batch_size, input_size)) for _ in range(nums_batch)]
labels = [paddle.randn((batch_size, output_size)) for _ in range(nums_batch)]

mse = paddle.nn.MSELoss()

model = SimpleNet(input_size, output_size)

optimizer = paddle.optimizer.SGD(learning_rate=0.0001, parameters=model.parameters())

start_timer()

for epoch in range(epochs):
   datas = zip(train_data, labels)
   for i, (data, label) in enumerate(datas):

      output = model(data)
      loss = mse(output, label)

      loss.backward()

      optimizer.step()
      optimizer.clear_grad()

print(loss)
end_timer_and_print("using defaule mode:")

model = SimpleNet(input_size, output_size)

optimizer = paddle.optimizer.SGD(learning_rate=0.0001, parameters=model.parameters())

# Step1：define GradScaler to scale the loss avoiding float overflow
scaler = paddle.amp.GradScaler(init_loss_scaling=1024)

start_timer()

for epoch in range(epochs):
   datas = zip(train_data, labels)
   for i, (data, label) in enumerate(datas):
      # Step2: create the context for AMP to start auto mixed precision training
      with paddle.amp.auto_cast():
            output = model(data)
            loss = mse(output, label)
      # Step3: use the GradScaler defined in Step1 to scale the loss and use the loss to back propagation
      scaled = scaler.scale(loss)
      scaled.backward()

      scaler.minimize(optimizer, scaled)
      optimizer.clear_grad()

print(loss)
end_timer_and_print("using AMP mode:")
