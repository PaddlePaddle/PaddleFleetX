# -*- coding: utf-8 -*-
from __future__ import print_function
from concurrent import futures
import grpc
from . import barrier_server_pb2 as barrier_server_pb2
from . import barrier_server_pb2_grpc as barrier_server_pb2_grpc
from concurrent import futures
from multiprocessing import Process

import time
import sys
import os


class BarrierClient(object):
    def __init__(self):
        self.current_endpoint = ""
        self.server_endpoint_ = ""
        self.stub = None

    @property
    def my_endpoint(self):
        return self.current_endpoint

    @my_endpoint.setter
    def my_endpoint(self, endpoint):
        self.current_endpoint = endpoint

    @property
    def server_endpoint(self):
        return self.server_endpoint_

    @server_endpoint.setter
    def server_endpoint(self, endpoint):
        self.server_endpoint_ = endpoint

    def connect(self):
        if self.server_endpoint_ == "" or self.current_endpoint == "":
            print("you should set server_endpoint and current_endpoint first")
            return
        while True:
            print("try to connect {}".format(self.server_endpoint_))
            options = [('grpc.max_message_length', 1024 * 1024 * 1024),
                       ('grpc.max_receive_message_length', 1024 * 1024 * 1024)]
            channel = grpc.insecure_channel(
                self.server_endpoint_, options=options)
            self.stub = barrier_server_pb2_grpc.BarrierServerStub(channel)
            try:
                req = barrier_server_pb2.Request()
                res = self.stub.SayHello(req)
                print("passed")
                break
            except:
                print("not ready")
                continue

        print("connected")

    def barrier(self):
        if self.stub == None:
            print("you should call connect() first")
            return

        while True:
            req = barrier_server_pb2.Request()
            req.endpoint = self.current_endpoint
            call_future = self.stub.ReadyToPass.future(req)
            req_res = call_future.result()
            if req_res.res_code == 0:
                print("pass")
                break
            else:
                time.sleep(0.5)
        return

    def close_server(self):
        if self.stub == None:
            print("you should call connect() first")
            return

        req = barrier_server_pb2.Request()
        call_future = self.stub.Exit.future(req)
        req_res = call_future.result()
