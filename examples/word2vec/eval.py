import paddle
import paddle.fluid as fluid
from conf import *
import logging
import six
import numpy as np
from network import word2vec_infer_net 

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("fluid")
logger.setLevel(logging.INFO)

def native_to_unicode(s):
    if _is_unicode(s):
        return s
    try:
        return _to_unicode(s)
    except UnicodeDecodeError:
        res = _to_unicode(s, ignore_errors=True)
        return res

def _is_unicode(s):
    if six.PY2:
        if isinstance(s, unicode):
            return True
    else:
        if isinstance(s, str):
            return True
    return False

def _to_unicode(s, ignore_errors=False):
    if _is_unicode(s):
        return s
    error_mode = "ignore" if ignore_errors else "strict"
    return s.decode("utf-8", errors=error_mode)

def _replace_oov(original_vocab, line):
    return u" ".join([
        word if word in original_vocab else u"<UNK>" for word in line.split()
    ])

def strip_lines(line, vocab):
    return _replace_oov(vocab, native_to_unicode(line))

def BuildWord_IdMap(dict_path):
    word_to_id = dict()
    id_to_word = dict()

    with open(dict_path, 'r') as f:
        for line in f:
            word_to_id[line.split(' ')[0]] = int(line.split(' ')[1])
            id_to_word[int(line.split(' ')[1])] = line.split(' ')[0]
    return word_to_id, id_to_word

def reader_creator(file_dir, word_to_id):
    def reader():
        files = os.listdir(file_dir)
        for fi in files:
            with open(file_dir + '/' + fi, "r") as f:
                for line in f:
                    if ':' in line:
                        pass
                    else:
                        line = strip_lines(line.lower(), word_to_id)
                        line = line.split()
                        yield [word_to_id[line[0]]], [word_to_id[line[1]]], [
                            word_to_id[line[2]]
                        ], [word_to_id[line[3]]], [
                            word_to_id[line[0]], word_to_id[line[1]],
                            word_to_id[line[2]]
                        ]
    return reader

def test(test_dir, w2i):
    return reader_creator(test_dir, w2i)

def prepare_data(file_dir, dict_path, batch_size):
    w2i, i2w = BuildWord_IdMap(dict_path)
    vocab_size = len(i2w)
    reader = paddle.batch(test(file_dir, w2i), batch_size)
    return vocab_size, reader, i2w

def eval_main(model_dir):
    vocab_size, test_reader, id2word = prepare_data(test_files_path, infer_dict_path, batch_size=infer_batch_size)
    
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
   
    startup_program = fluid.framework.Program()
    test_program = fluid.framework.Program()
    
    with fluid.framework.program_guard(test_program,startup_program):
        values, pred = word2vec_infer_net(vocab_size, embedding_size)
    
        fluid.io.load_persistables(
            executor=exe, dirname=model_dir, main_program=fluid.default_main_program())
    
        accum_num = 0
        accum_num_sum = 0.0
        step_id = 0
        for data in test_reader():
            step_id += 1
            b_size = len([dat[0] for dat in data])
            wa = np.array([dat[0] for dat in data]).astype("int64").reshape(b_size, 1)
            wb = np.array([dat[1] for dat in data]).astype("int64").reshape(b_size, 1)
            wc = np.array([dat[2] for dat in data]).astype("int64").reshape(b_size, 1)
    
            label = [dat[3] for dat in data]
            input_word = [dat[4] for dat in data]
            para = exe.run(fluid.default_main_program(),
                           feed={
                               "analogy_a": wa, "analogy_b": wb, "analogy_c": wc,
                               "all_label": np.arange(vocab_size).reshape(vocab_size, 1).astype("int64"),
                           },
                           fetch_list=[pred.name, values],
                           return_numpy=False)
            pre = np.array(para[0])
            val = np.array(para[1])
            for ii in range(len(label)):
                top4 = pre[ii]
                accum_num_sum += 1
                for idx in top4:
                    if int(idx) in input_word[ii]:
                        continue
                    if int(idx) == int(label[ii][0]):
                        accum_num += 1
                    break
            if step_id % 1 == 0:
                logger.info("step:%d %d " % (step_id, accum_num))
        acc = 1.0 * accum_num / accum_num_sum
        logger.info("acc:%.3f " % acc)
        return acc

if __name__ == '__main__':
    models = os.listdir(model_path)
    infer_result = {}
    for model in models:
        epoch = model.split('_')[-1]
        logger.info("process %s" % model)
        res = eval_main(os.path.join(model_path, model))
        infer_result[int(epoch)] = res
