import paddle
import numpy as np
__all__ = []
if paddle.is_compiled_with_rocm():
    # There is a bug in matmul case 13 (batchsize>1) on DCU in fp16 mode, (maybe too large).
    # Here split the big matrix to few small matrices with batchsize=1 (matmul case 8) to compute the result.
    # TODO: fix the bug in matmul kernel
    def rocm_matmul(x, y, *args, **kwargs):
        if  (x.dtype != paddle.float16 and y.dtype != paddle.float16) or \
            len(x.shape) < 3 or len(y.shape) < 3 or \
            np.prod(x.shape[:-2]) == 1:
            return _matmul(x, y, *args, **kwargs)

        w_ = paddle.reshape(x, (-1, x.shape[-2], x.shape[-1]))
        v_ = paddle.reshape(y, (-1, y.shape[-2], y.shape[-1]))
        out_=[]
        for (tw, tv) in zip(w_, v_):
            out_.append(_matmul(tw, tv, *args, **kwargs))
        out_ = paddle.stack(out_)
        out_ = paddle.reshape(out_, x.shape[:-2]+out_.shape[-2:])
        return out_

    if not paddle.matmul == rocm_matmul:
        _matmul = paddle.matmul
    setattr(paddle, "matmul", rocm_matmul)
    __all__ += [ "rocm_matmul" ]


if paddle.is_compiled_with_xpu():
    pass

if paddle.is_compiled_with_npu():
    pass
