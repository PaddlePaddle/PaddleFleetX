import paddle

def get_device_and_mapping():
    """
        Return device type and name-bool mapping implifying which type is supported.
    """
    suppoted_device_map = {
        "gpu": paddle.is_compiled_with_cuda(),
        "xpu": paddle.is_compiled_with_xpu(),
        "rocm": paddle.is_compiled_with_rocm(),
        "npu": paddle.is_compiled_with_npu(),
        "cpu": True
    }
    for d, v in suppoted_device_map.items():
        if v:
            return d, suppoted_device_map


def get_device():
    """
        Return the device with which the paddle is compiled, including 'gpu'(for rocm and gpu), 'npu', 'xpu', 'cpu'.
    """
    d, _ = get_device_and_mapping()
    return d


def synchronize():
    """
    Synchronize device, return True if succeeded, otherwise return False
    """
    if paddle.is_compiled_with_cuda():
        paddle.device.cuda.synchronize()
        return True
    elif paddle.is_compiled_with_xpu():
        paddle.device.xpu.synchronize()
        return True
    else:
        logger.warning("The synchronization is only supported on cuda and xpu now.")
    return False
