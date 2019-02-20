/***************************************************************************
 * 
 * Copyright (c) 2018 Baidu.com, Inc. All Rights Reserved
 * $Id$ 
 * 
 **************************************************************************/
#include "common/afs_warpper.h"
#include "common/shell.h"
#include "gtest/gtest.h" 
#include <gflags/gflags.h>

using namespace paddle::ps;
DEFINE_string(pslib_afs_test_hadoop_bin, "/home/work/local/hadoop-client-1.5.9/hadoop/bin/hadoop", "path of hadoop");

TEST(AfsClient, common_operation) {
    shell_verbose_internal() = true;
    ::paddle::FsClientParameter fs_client_config;
    fs_client_config.set_user("mlarch_rd");
    fs_client_config.set_passwd("Fv1M87");
    fs_client_config.set_uri("afs://xingtian.afs.baidu.com:9902");
    fs_client_config.set_hadoop_bin(FLAGS_pslib_afs_test_hadoop_bin);

    
    AfsClient afs_client;
    afs_client.initialize(fs_client_config);
    
    //remove-all
    afs_client.remove_dir("afs:/user/mlarch_rd/xiexionghang/pslib/test");
    //write
    FsChannelConfig channel_config;
    channel_config.path = "afs:/user/mlarch_rd/xiexionghang/pslib/test/afs_test.txt";
    channel_config.converter = "(grep 12)";
    auto channel_w = afs_client.open_w(channel_config);
    ASSERT_EQ(channel_w->write_line("1234"), 0);
    ASSERT_EQ(channel_w->write_line("2234"), 0);
    ASSERT_EQ(channel_w->write_line("3234"), 0);
    channel_w->close();

    //list
    auto fs_list = afs_client.list("afs:/user/mlarch_rd/xiexionghang/pslib/test/");
    ASSERT_EQ(fs_list.size(), 1);
    ASSERT_EQ(fs_list[0], channel_config.path);
    
    //read
    std::string line_data;
    auto channel_r = afs_client.open_r(channel_config);
    ASSERT_EQ(channel_r->read_line(line_data), 0);
    ASSERT_EQ(line_data, "1234");
    ASSERT_EQ(channel_r->read_line(line_data), 0);
    ASSERT_EQ(line_data, "2234");
    ASSERT_EQ(channel_r->read_line(line_data), 0);
    ASSERT_EQ(line_data, "3234");
    channel_r->close();
}



















/* vim: set ts=4 sw=4 sts=4 tw=100 */
