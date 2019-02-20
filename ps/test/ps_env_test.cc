/***************************************************************************
 * 
 * Copyright (c) 2018 Baidu.com, Inc. All Rights Reserved
 * $Id$ 
 * 
 **************************************************************************/
#include "communicate/ps_env.h"
#include "gtest/gtest.h" 
#ifdef MPI_TRAIN_ENV
#include <mpi.h>
#endif

using namespace paddle::ps;

TEST(PSEnv, pshost) {
    PSHost host;
    host.ip = "127.0.0.1";
    host.port = 1103;
    host.rank = 0;
    uint64_t host_sign = host.serialize_to_uint64();

    PSHost host2;
    host2.parse_from_uint64(host_sign);
    ASSERT_EQ(host2.ip, host.ip);
    ASSERT_EQ(host2.port, host.port);
    ASSERT_EQ(host2.rank, host.rank);
#ifdef MPI_TRAIN_ENV
    int mpi_argc = 0;
    char** mpi_argv = NULL;
    std::cout << "test mpi env gather\n";
    MPI_Init(&mpi_argc, &mpi_argv);
    MpiPSEnvironment mpi_env;
    mpi_env.registe_ps_host(host.ip, host.port, host.rank);
    ASSERT_EQ(mpi_env.gather_ps_hosts(), 0);
    auto ps_server_list = mpi_env.get_ps_hosts();
    ASSERT_EQ(ps_server_list.size(), 1);
    ASSERT_EQ(ps_server_list[0].ip, host.ip);
#endif
}



















/* vim: set ts=4 sw=4 sts=4 tw=100 */
