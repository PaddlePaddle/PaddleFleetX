import sys
from barrier_server_impl import BarrierServer

server = BarrierServer()
server_endpoint = sys.argv[1]
worker_endpoints = sys.argv[2].split("-")
server.start_server_foreground(endpoint=server_endpoint,
                               worker_endpoints=worker_endpoints)

