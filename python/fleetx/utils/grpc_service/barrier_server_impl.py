# -*- coding: utf-8 -*-
from __future__ import print_function
from concurrent import futures
import grpc
import fleetx.utils.grpc_service.barrier_server_pb2 as barrier_server_pb2
import fleetx.utils.grpc_service.barrier_server_pb2_grpc as barrier_server_pb2_grpc
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

    def Exit(self, request, context):
        with open("_shutdown_barrier_server", "w") as f:
            f.write("_shutdown_barrier_server\n")
        res = barrier_server_pb2.Res()
        res.res_code = 0
        return res


class BarrierServer(object):
    def __init__(self):
        pass

    def start(self,
              max_workers=1000,
              concurrency=100,
              endpoint="",
              worker_endpoints=[]):
        if endpoint == "":
            logging.info("You should specify endpoint in start function")
            return
        server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=max_workers),
            options=[('grpc.max_send_message_length', 1024 * 1024 * 1024),
                     ('grpc.max_receive_message_length', 1024 * 1024 * 1024)],
            maximum_concurrent_rpcs=concurrency)
        barrier_server_pb2_grpc.add_BarrierServerServicer_to_server(
            BarrierServerServicer(worker_endpoints), server)
        server.add_insecure_port('[::]:{}'.format(endpoint))
        server.start()
        os.system("rm _shutdown_barrier_server")
        while (not os.path.isfile("_shutdown_barrier_server")):
            time.sleep(1)


if __name__ == "__main__":
    barrier_server = BarrierServer()
    barrier_server.start(
        endpoint=sys.argv[1], worker_endpoints=sys.argv[2].split("-"))
    #barrier_server.start(endpoint=sys.argv[1], worker_endpoints=sys.argv[2])
