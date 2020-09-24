# -*- coding: utf-8 -*-
from __future__ import print_function
from concurrent import futures
import grpc
from . import barrier_server_pb2 as barrier_server_pb2
from . import barrier_server_pb2_grpc as barrier_server_pb2_grpc
from .barrier_client_impl import BarrierClient
import time
import sys, os
import logging


class BarrierServerServicer(object):
    def __init__(self, worker_endpoints):
        self.endpoint_list = worker_endpoints
        self.barrier_lock = True  # if true, barrier is on
        self.barrier_set = set()
        self.unbarrier_set = set()

    def ReadyToPass(self, request, context):
        req_str = request.endpoint
        res = barrier_server_pb2.Res()
        if len(self.barrier_set) == len(self.endpoint_list):
            res.res_code = 0
            self.unbarrier_set.add(req_str)
            if len(self.unbarrier_set) == len(self.endpoint_list):
                self.barrier_set = set()
                self.unbarrier_set = set()
        else:
            self.barrier_set.add(req_str)
            res.res_code = 1
        return res

    def SayHello(self, request, context):
        res = barrier_server_pb2.Res()
        res.res_code = 0
        return res

    def Exit(self, request, context):
        with open("_shutdown_barrier_server", "w") as f:
            f.write("_shutdown_barrier_server\n")
        res = barrier_server_pb2.Res()
        res.res_code = 0
        return res


class BarrierServer(object):
    def __init__(self):
        self.p = None

    def _start_func(self, port, worker_endpoints):
        barrier_server = BarrierServer()
        barrier_server._start(endpoint=port, worker_endpoints=worker_endpoints)

    def start_server_in_background(self,
                                   max_workers=1000,
                                   concurrency=100,
                                   endpoint="",
                                   worker_endpoints=[]):
        if endpoint == "":
            logging.info("You should specify endpoint in start function")
            return
        self.server_endpoint = endpoint
        self.worker_endpoints = worker_endpoints
        from multiprocessing import Process
        self.p = Process(
            target=self._start_func, args=(endpoint, worker_endpoints))
        self.p.start()

    def close_server(self):
        client = BarrierClient()
        client.server_endpoint = self.server_endpoint
        client.my_endpoint = "127.0.0.1:4444"  # this is not useful in this func
        client.connect()
        client.close_server()
        self.p.join()

    def start_server_foreground(self,
                                max_workers=1000,
                                concurrency=100,
                                endpoint="",
                                worker_endpoints=[]):
        self._start(max_workers, concurrency, endpoint, worker_endpoints)

    def _start(self,
               max_workers=1000,
               concurrency=100,
               endpoint="",
               worker_endpoints=[]):
        if endpoint == "":
            logging.info("You should specify endpoint in start function")
            return

        self.server_endpoint = endpoint
        self.server_port = endpoint.split(":")[1]
        self.worker_endpoints = worker_endpoints

        server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=max_workers),
            options=[('grpc.max_send_message_length', 1024 * 1024 * 1024),
                     ('grpc.max_receive_message_length', 1024 * 1024 * 1024)],
            maximum_concurrent_rpcs=concurrency)
        barrier_server_pb2_grpc.add_BarrierServerServicer_to_server(
            BarrierServerServicer(self.worker_endpoints), server)
        server.add_insecure_port('[::]:{}'.format(self.server_port))
        server.start()
        print("Server started")
        os.system("rm _shutdown_barrier_server")
        while (not os.path.isfile("_shutdown_barrier_server")):
            time.sleep(1)


if __name__ == "__main__":
    barrier_server = BarrierServer()
    server_endpoint = sys.argv[1]
    worker_endpoints = sys.argv[2].split("-")
    barrier_server.start_server_in_background(
        endpoint=server_endpoint, worker_endpoints=worker_endpoints)

    def barrier_client(my_endpoint):
        client = BarrierClient()
        client.server_endpoint = server_endpoint
        client.my_endpoint = my_endpoint
        client.connect()
        client.barrier()

    from multiprocessing import Process
    import time
    p_list = []
    for ep in worker_endpoints:
        p = Process(target=barrier_client, args=(ep, ))
        p.start()
        p_list.append(p)
        time.sleep(5)

    for p in p_list:
        p.join()

    barrier_server.close_server()
