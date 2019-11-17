import paddle.fluid as fluid

class InputMaker(object):
    def __init__(self):
        pass

    def sparse(self, name=None):
        return fluid.layers.data(
            name=name, shape=[1], lod_level=1, dtype='int64')
    
    def dense(self, name=None, shape=None):
        return fluid.layers.data(
            name=name, shape=shape, dtype='int64')

