import os

def is_first_worker():
    PADDLE_TRAINER_ENDPOINTS = os.environ.get('PADDLE_TRAINER_ENDPOINTS')
    if PADDLE_TRAINER_ENDPOINTS is None:
        return True
    endpoints = PADDLE_TRAINER_ENDPOINTS.split(",")
    current_endpoint = os.environ.get('PADDLE_CURRENT_ENDPOINT')
    hostname, _ = current_endpoint.split(":")
    host_endpoints = [x for x in endpoints if x.split(":")[0] == hostname]
    return host_endpoints[0] == current_endpoint

