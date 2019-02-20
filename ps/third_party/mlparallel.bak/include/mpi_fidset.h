/**************************************************************************
 *
 * Copyright (c) 2010 Baidu.com, Inc. All Rights Reserved
 *
 *************************************************************************/

/**
* @file mpi_fidset.h
* @author wuzhihua02(wuzhihua02@baidu.com)
* @date 2015/1/15 15:50:00
* @version $Revision$
* @brief data struct to store word ids
*
**/

#ifndef MPIWRAPPER_MPI_FIDSET_H
#define MPIWRAPPER_MPI_FIDSET_H
#include <set>
#include <algorithm>
#include <vector>
#include <pthread.h>
#include "mpi_common.h"

namespace mpi_wrapper {

/**
 * @brief Basic struct to store word ids for one sparse.
 */
class FidSet {
public:
    /**
     * @brief constructor for feature id set.
    **/
    FidSet();

    /**
     * @brief destructor for feature id set.
    **/
    ~FidSet();

    /**
     * @brief allocate space to init data structure.
     * @return int - 0: success; -1: fail
    **/
    int init();

    /**
     * @brief insert one word id into set
     * @param [in] wordid : int - one word id
     * @return void
    **/
    void insert(int wordid);

    /**
     * @brief insert word array into set
     * @param [in] arr_start_ptr : int* - start of wordid array
     * @param [in] arr_end_ptr   : int* - end of wordid array
     * @return void
    **/
    void insert_list(int* arr_start_ptr, int* arr_end_ptr);

    /**
     * @brief get the number of wordids
     * @return int - num of wordids
    **/
    int fid_size();

    /**
     * @brief get the number of wordids
     * @return int - num of wordids
    **/
    int fidold_size();

    /**
     * @brief merge wordids for all threads into old fidset
     * @param [out] fidset : const FidSet&
     * @return void
    **/
    void union_fidold(const FidSet& fidset);

    /**
    * @brief get wordids from old fidset
    * @param [out] vec : std::vector<int>&
    * @return int - the number of wordids
    **/
    int get_fidold_value(std::vector<int>& vec);

    /**
    * @brief swap fidset and old fidset
    * @return void
    **/
    void swap();

    /**
    * @brief clear old fidset
    * @return void
    **/
    void clear_fidold();

    /**
    * @brief clear fidset
    * @return void
    **/
    void clear_fid();

#ifdef GPU
    /**
    * @brief get wordids
    * @return std::set<int>* - wordids
    **/
    std::set<int>* get_fidset();
#endif

private:
    // fidset for insert
    std::set<int>* _fid;

    // old fidset for union and send
    std::set<int>* _fid_old;

    // lock for fidset insert
    pthread_spinlock_t _lock;
};

/**
 * @brief class to store workers' each sparse data's changed keys.
 *
 */
class FidSetParam {
public:
    /**
     * @brief constructor for feature id set - multi sparse, thread.
    **/
    FidSetParam();

    /**
     * @brief destructor for feature id set - multi sparse, thread.
    **/
    ~FidSetParam();

    /**
     * @brief get sparse number
     * @return int - sparse number
    **/
    int get_sparse_num();

    /**
     * @brief init key parameter for sparse data; for each sparse data, there is a fidset.
     * @param [in] sparse_num : int - sparse number
     * @param [in] thread_num : int - thread number
     * @return int - 0: success; -1: error
    **/
    int init(int sparse_num, int thread_num);

    /**
     * @brief copy data from new to old and merge all data to 0
     * @return void
    **/
    void union_fidset();

    /**
     * @brief insert key list for one sparse's one thread.
     * @param [in] sparse_index : int - sparse index
     * @param [in] thread_index : int - thread index
     * @param [in] key_num      : int - wordid number
     * @param [in] key_ids      : int*- wordid array
     * @return int - 0: success; -1: error
    **/
    int add_fidset(int sparse_index, int thread_index, int key_num, int* key_ids);

    /**
     * @brief get one sparse's all key data with clear.
     * @param [in] sparse_index : int - sparse index
     * @param [out] wordid       : std::vector<int>& wordid
     * @return int - word id number
    **/
    int get_fidset(int sparse_index, std::vector<int>& wordid);

    /**
     * @brief clear data - reset all old keys to no change 0.
     * @return void
    **/
    void clear_old();

    /**
     * @brief clear old fidset and fidset
     * @return void
    **/
    void clear_all();

#ifdef GPU
    /**
     * @brief query one sparse's all key data without clear
     * @param [in] sparse_index : int - sparse index
     * @param [out] wordid       : std::vector<int>& - word id vector
     * @return int - word id number
    **/
    int get_fids_noclear(int sparse_index, std::vector<int>& wordid);

    /**
     * @brief query one sparse's all key data without clear
     * @param [in] sparse_index : int - sparse index
     * @param [in] arr_len      : int - array capacity
     * @param [out] arr         : int*- array
     * @param [out] size        : int - word number
     * @return int - word id number
    **/
    int get_fids_noclear(int sparse_index, size_t arr_len, int* arr, int& size);
#endif

private:
    // sparse number
    int _sparse_num;

    // thread number
    int _thread_num;

    // fidset for multi sparse and thread
    FidSet* _sparse[MAX_SPARSE_NUM];

#ifdef GPU
    // fidset for all stored fidids
    std::set<int>* _fid_cur_all;
#endif
};

}

#endif
