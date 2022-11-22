import nvidia_smi

nvidia_smi.nvmlInit()

handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
# card id 0 hardcoded here, there is also a call to get all available card ids, so we could iterate

info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)

print("Total memory:", info.total/1024**3, "GB")
print("Free memory:", info.free/1024**3, "GB")
print("Used memory:", info.used/1024**3, "GB")

nvidia_smi.nvmlShutdown()
