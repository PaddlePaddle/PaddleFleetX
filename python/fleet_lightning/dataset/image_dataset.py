import paddle.fluid as fluid

def image_dataset_from_filelist(filelist, inputs, batch_size=32,
                                shuffle=True, phase="train"):
    def reader():
        yield None
    loader = fluid.io.DataLoader.from_generator(
        inputs, capacity=16, iterable=ITERABLE)
    loader.set_batch_generator(
        fluid.io.batch(reader ,batch_size=32), fluid.cpu_places())
    return loader
