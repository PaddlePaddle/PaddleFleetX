"""
Reader using visreader.
"""
from __future__ import division
from __future__ import print_function
from visreader.reader_builder import ReaderBuilder
from visreader.reader_builder import ReaderSetting

THREAD = 8


def _parse_kv(r):
    """
    parse kv data from sequence file for imagenet dataset.
    """
    import cPickle
    k, v = r
    obj = cPickle.loads(v)
    return obj['image'], obj['label']


def train(data_dir, num_threads=THREAD):
    args = dict()
    args['worker_mode'] = 'python_thread'
    args['use_sharedmem'] = False
    args['worker_num'] = num_threads
    settings = {
        'sample_parser': _parse_kv,
        'lua_fname': None,
        'worker_args': args
    }

    train_setting = ReaderSetting(
        data_dir, sc_setting={'pass_num': 1}, pl_setting=settings)
    settings = {'train': train_setting}
    rd_builder = ReaderBuilder(settings=settings, pl_name='imagenet')
    train_reader = rd_builder.train()
    return train_reader 
