import paddle.fluid as fluid

input_x = fluid.layers.data()
input_y = fluid.layers.data()

fc_1 = fluid.layers.fc(input=input_x)
fc_2 = fluid.layers.fc(input=fc_1)
prediction = fluid.layers.fc(input=[fc_2])
cost = fluid.layers.cross_entropy(input=prediction, label=input_y)
optimizer = fluid.optimizer.Adagrad(learning_rate=0.01)
optimizer.minimize(cost)
place = fluid.CPUPlace()

pass_num = 10
for pass_id in range(pass_num):
    for x in xx():
        

