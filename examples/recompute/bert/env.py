import os


def dist_env():
    '''
    Return a dict of all variable that distributed training may use.
    NOTE: you may rewrite this function to suit your cluster environments.
    '''
    default_port = "8701"
    port = os.getenv('PADDLE_PORT', default_port)

    worker_endpoints = []
    trainer_endpoints = ''
    if os.getenv('PADDLE_TRAINER_ENDPOINTS'):
        trainer_endpoints = os.getenv('PADDLE_TRAINER_ENDPOINTS')
    else:# for paddlecloud
        worker_ips = os.getenv('PADDLE_TRAINERS', '')
        for ip in worker_ips.split(','):
            worker_endpoints.append(':'.join([ip, port]))
        trainer_endpoints = ','.join(worker_endpoints)

    current_endpoint = ''
    if os.getenv('PADDLE_CURRENT_ENDPOINT'):
        current_endpoint = os.getenv('PADDLE_CURRENT_ENDPOINT')
    else:# for paddlecloud
        current_endpoint = os.getenv('POD_IP', '') + ':' + port

    trainer_id = int(os.getenv('PADDLE_TRAINER_ID', '0'))
    num_trainers = 1
    if trainer_endpoints:
        num_trainers = len(trainer_endpoints.split(','))
    
    return {
        'trainer_id': trainer_id,
        'num_trainers': num_trainers,
        'current_endpoint': current_endpoint,
        'trainer_endpoints': trainer_endpoints
    }

