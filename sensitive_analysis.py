import pickle
import matplotlib.pyplot as plt
import numpy as np

def read_from_pickle(path):
    with open(path, 'rb') as file:
        try:
            while True:
                yield pickle.load(file)
        except EOFError:
            pass

def plot(data):
    ratio = 0.05
    new_data = []
    minimum = 0.0
    maximum = 0.0
    while ratio < 0.3:
        ratio = round(ratio, 2)
        cur_x = []
        for key, val in data.items():
            if key == "linear_2.w_0": continue
            loss = val.get(ratio)
            print(ratio, key, loss)
            assert loss is not None
            cur_x.append(loss)
        minimum = min(minimum, min(cur_x))
        maximum = max(maximum, max(cur_x))
        new_data.append(cur_x)
        ratio += 0.05

    width = 0.1
    x = np.arange(len(data)-1)
 
    # plt.ylim(minimum, maximum)

    # plot data in grouped manner of bar type
    plt.bar(x-0.2, new_data[0], width, color='blue')
    plt.bar(x-0.1, new_data[1], width, color='cyan')
    plt.bar(x, new_data[2], width, color='orange')
    plt.bar(x+0.1, new_data[3], width, color='green')
    plt.bar(x+0.2, new_data[4], width, color='red')

    xticks = ["{}".format(i*4+2) for i in range(1, len(data))]
    plt.xticks(x, xticks)
    plt.xlabel("layer-names")
    plt.ylabel("loss percent")
    plt.legend(["ratio=0.05", "ratio=0.10", "ratio=0.15", "ratio=0.20", "ratio=0.25"])
    plt.show()
    plt.savefig('loss-percent.png')


iterator = read_from_pickle('sen.pickle')
for i in iterator:
    sens = i.sensitivies
plot(sens)
