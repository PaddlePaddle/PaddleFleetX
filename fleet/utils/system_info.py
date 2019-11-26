import cup
from cup.res import linux
from threading import Thread
import time

class CustomThread(Thread):
    def __init__(self, func, args=()):
        super(CustomThread, self).__init__()
        self.func = func
        self.args = args

    def run(self):
        self.result = self.func(*self.args)

    def get_result(self):
        Thread.join(self)
        try:
            return self.result
        except Exception:
            return None

def get_cpu_usage(interval=10, counter=100):
    cpu_list = []
    c = 0
    while c < counter:
        cpuinfo = linux.get_cpu_usage(
            intvl_in_sec=interval)
        cpu_list.append(cpuinfo.usr)
        c += 1
    return cpu_list

def get_mem_usage(interval=10, counter=100):
    mem_list = []
    c = 0
    while c < counter:
        meminfo = linux.get_meminfo()
        mem_list.append((meminfo.total - meminfo.available) / 1024 / 1024 / 1024)
        c += 1
    return mem_list

def get_net_usage(interval=10, counter=100):
    net_list = []
    c = 0
    while c < counter:
        net_in, net_out = linux.get_net_through(str_interface="eth0")
        net_list.append((net_in, net_out))
        c += 1
    return net_list

def get_net_recv_speed(interval=10, counter=100):
    net_recv_list = []
    c = 0
    while c < counter:
        net_recv = linux.get_net_recv_speed(
            str_interface="eth0", intvl_in_sec=interval)
        net_recv_list.append(net_recv)
        c += 1
    return net_recv_list

def get_net_send_speed(interval=10, counter=100):
    net_send_list = []
    c = 0
    while c < counter:
        net_send = linux.get_net_transmit_speed(
            str_interface="eth0", intvl_in_sec=interval)
        net_send_list.append(net_send)
        c += 1
    return net_send_list

def get_system_info():
    # total memory
    # total cpu num
    # kernel version
    sys_info = {}
    linux_info = linux.get_meminfo()
    sys_info["total_memory"] = linux_info.total
    sys_info["cpu_num"] = linux.get_cpu_nums()
    sys_info["kernel"] = " ".join(linux.get_kernel_version())
    from cup import net
    sys_info["ip_addr"] = net.getip_byinterface('eth0')

    return sys_info

def launch_system_monitor(interval, count_num):
    task_info = [[get_cpu_usage, interval, count_num],
                 [get_mem_usage, interval, count_num],
                 [get_net_usage, interval, count_num],
                 [get_net_recv_speed, interval, count_num],
                 [get_net_send_speed, interval, count_num]]
    results = []
    threads = []
    for ti in task_info:
        thr = CustomThread(ti[0], args=(ti[1], ti[2],))
        threads.append([thr, ti[0].__name__])
    #return pool, results
    for thr in threads:
        thr[0].start()
    return threads

def get_monitor_result(threads):
    result_dict = {}
    for thr in threads:
        result = thr[0].get_result()
        key = thr[1]
        result_dict[key] = result
    return result_dict

if __name__ == "__main__":
    threads = launch_system_monitor(4, 10)
    #res_dict = get_system_info()
    get_monitor_result(threads)
    
    
