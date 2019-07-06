import pre_sampled_pretraining as psp
import sys

data_dir = sys.argv[1]
batch_size = int(sys.argv[2])
in_tokens = False
epoch=1
reader = psp.DataReader(data_dir=data_dir, batch_size=batch_size,
                        in_tokens=in_tokens, epoch=epoch)

data_gen = reader.data_generator()
for data in data_gen():
    for x in data:
        print(x.shape)
