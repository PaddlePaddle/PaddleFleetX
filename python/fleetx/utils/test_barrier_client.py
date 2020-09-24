import sys
from barrier_client_impl import BarrierClient

client = BarrierClient()
server_endpoint = sys.argv[1]
worker_endpoints = sys.argv[2].split("-")
index = sys.argv[3]

client.server_endpoint = server_endpoint
client.my_endpoint = worker_endpoints[int(index)]
client.connect()
client.barrier()

if index == "0":
    client.close_server()
