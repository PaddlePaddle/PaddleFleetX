/***************************************************************************
 *
 * Copyright (c) 2015 Baidu.com, Inc. All Rights Reserved
 * $Id$
 *
 **************************************************************************/

/**
* @file mpi_util.h
* @author wuzhihua02(wuzhihua02@baidu.com)
* @date 2015/1/15 15:50:00
* @version $Revision$
* @brief
*
**/
#ifndef MPIWRAPPER_MPI_UTIL_H
#define MPIWRAPPER_MPI_UTIL_H

#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <string>
#include <vector>
#include <fstream>
#include <new>
#include "mpi_timer.h"
#include "mpi_model_param.h"
namespace mpi_wrapper {

//TODO: remove it
/**
 * @brief struct for train files
 */
struct trainfile_t {
    char name[MAX_STRING_LEN];
};

/**
 * @brief struct for download info in multithread
 */
struct download_thread_info_t {
    // thread id
    int thread_id;

    // start index
    int start;

    // end index
    int end;

    // total downloaded file number
    int* pnum;

    // total thread number
    int download_data_thread_num;

    // train files hadoop path
    std::string files_dir;

    // output folder for downloaded files
    std::string out_folder;

    // array for downloaded files
    trainfile_t* files_array;
};

class ThreadSync {
public:
    ThreadSync();

    int init(int thread);

    ~ThreadSync();

    void thread_sync();

private:

    pthread_barrier_t _pbarrier;
};

class ThreadLock {
public:
    ThreadLock();

    ~ThreadLock();

    void thread_wait_true();

    void thread_lock();

    void thread_unlock();

    void thread_set(bool value);

private:
    volatile bool _flag_value;

    pthread_spinlock_t _lock;
};

/**
 * @brief check system reture value
 * @param [in] status : int - return value from system()
 * @return bool : true-success; false-error
 **/
bool check_system_ret(int status);

int exec_system_cmd(std::string);

/**
 * @brief split string by separator with max column limitation
 * @param [in] in_str   : const std::string& - input string
 * @param [in] separator: char - separator character
 * @param [in] col_limit: uint32_t - max column number
 * @param [out] result  : std::vector<std::string>& - output vector for split results
 * @return int : void
 **/
void split_string(const std::string& in_str,
                  char separator,
                  uint32_t col_limit,
                  std::vector<std::string>& result);

/**
 * @brief split string by separator
 * @param [in] in_str   : const std::string& - input string
 * @param [in] separator: char - separator character
 * @param [out] result  : std::vector<std::string>& - output vector for split results
 * @return void
 **/
void split_string(const std::string& in_str,
                  char separator,
                  std::vector<std::string>& result);
/**
 * @brief check whether local file exits
 * @param [in] file_name   : const char* - file name
 * @return bool : true-exits; false-not exit
 **/
bool is_file_exist(const char* file_name);

// TODO: move it to mpi_model_parameter or mpiwrapper
/**
 * @brief get total dense param real size
 * @param [in] model  : const ModelParam& model - model parameter
 * @return int : size
 **/
int get_denseparam_total_data_size(const ModelParam& model);

// TODO: simplify this
/**
 * @brief check whether the two input are equal or not
 * @param [in] a : bool - value a
 * @param [in] b : bool - value b
 * @return bool : true - equal; false - unequal;
 **/
bool is_equal(const bool a, const bool b);

/**
 * @brief check whether the two input are equal or not
 * @param [in] a : int - value a
 * @param [in] b : int - value b
 * @return bool : true - equal; false - unequal;
 **/
bool is_equal(const int a, const int b);

/**
 * @brief check whether the two input are equal or not
 * @param [in] a : float - value a
 * @param [in] b : float - value b
 * @return bool : true - equal; false - unequal;
 **/
bool is_equal(const float a, const float b);

/**
 * @brief check whether the two input are equal or not
 * @param [in] a : const std::string& - value a
 * @param [in] b : const std::string& - value b
 * @return bool : true - equal; false - unequal;
 **/
bool is_equal(const std::string& a, const std::string& b);
}
#endif  // MPI_UTIL_H

/* vim: set ts=4 sw=4 sts=4 tw=100 */
