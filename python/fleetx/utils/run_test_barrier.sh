python test_barrier_server.py 127.0.0.1:60005 127.0.0.1:60000-127.0.0.1:60001  &
python test_barrier_client.py 127.0.0.1:60005 127.0.0.1:60000-127.0.0.1:60001 0 &
python test_barrier_client.py 127.0.0.1:60005 127.0.0.1:60000-127.0.0.1:60001 1 &
