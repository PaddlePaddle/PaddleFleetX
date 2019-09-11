import os
import paddle.fluid as fluid


def nccl2_prepare(args, startup_program, main_program):
    # envs = args.dist_env
    trainer_id = int(os.getenv("PADDLE_TRAINER_ID", "0"))
    current_endpoint = os.getenv("PADDLE_CURRENT_ENDPOINT", "127.0.0.1:6170")
    trainer_endpoints = os.getenv("PADDLE_TRAINER_ENDPOINTS", "127.0.0.1:6170")

    config = fluid.DistributeTranspilerConfig()
    config.mode = "nccl2"
    if args.nccl_comm_num > 1:
        config.nccl_comm_num = args.nccl_comm_num
    trainers_num = len(trainer_endpoints.split(','))
    if args.use_hallreduce and trainers_num > 8:
        config.use_hierarchical_allreduce = args.use_hallreduce
        config.hierarchical_allreduce_inter_nranks = 8

        assert config.hierarchical_allreduce_inter_nranks > 1
        assert trainers_num % config.hierarchical_allreduce_inter_nranks == 0
        config.hierarchical_allreduce_exter_nranks = trainers_num / config.hierarchical_allreduce_inter_nranks

    t = fluid.DistributeTranspiler(config=config)

    t.transpile(
        trainer_id,
        trainers=trainer_endpoints,
        current_endpoint=current_endpoint,
        startup_program=startup_program,
        program=main_program)
