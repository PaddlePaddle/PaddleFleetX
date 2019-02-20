/***************************************************************************
 *
 * Copyright (c) 2015 Baidu.com, Inc. All Rights Reserved
 * $Id$
 *
 **************************************************************************/

/**
* @file mpi_fidset.h
* @author wuzhihua02(wuzhihua02@baidu.com)
* @date 2015/1/15 15:50:00
* @version $Revision$
* @brief data struct to store word ids
*
**/

#ifndef MPIWRAPPER_MPI_MODEL_PARAM_H
#define MPIWRAPPER_MPI_MODEL_PARAM_H

#include <new>
#include <vector>
#include <stdio.h>
#include <string.h>
#include "mpi_common.h"
namespace mpi_wrapper {

/**
 * @brief struct for dense info and dense data.
 *
 */
struct dense_t {
    // row number
    int row;

    // start pos for current process
    int cur_start_pos;

    // row number for current process
    int cur_row;

    // column number
    int col;

    // total mem size
    int size;

    // dense data
    real* pvalue;

    // old dense data
    real* pvalue_old;

    // diff for server
    real* pvalue_diff;

    /**
     * @brief construtor for dense data.
     */
    dense_t() {
        row = 0;
        cur_start_pos = 0;
        cur_row = 0;
        col = 0;
        size = 0;
        pvalue = NULL;
        pvalue_old = NULL;
        pvalue_diff = NULL;
    }
};

/**
 * @brief struct for sparse info and sparse data.
 */
struct sparse_t {
    // changed key number
    int changed_key_num;

    // dim for one word of this sparse.
    int sparse_dim;

    // total key number
    int key_num;

    // total mem size
    int size;

    // changed word ids
    int* pkey;

    // all parameters for all word ids
    real* pvalue;

    // old parameter for all word ids
    real* pvalue_old;

    // diff for server
    real* pvalue_diff;

    /**
     * @brief construtor for sparse data.
     */
    sparse_t() {
        changed_key_num = 0;
        sparse_dim = 0;
        key_num = 0;
        size = 0;
        pkey = NULL;
        pvalue = NULL;
        pvalue_old = NULL;
        pvalue_diff = NULL;
    }
};

/**
 * @brief information for dividing dense to dense-part;
 * each server has several dense-parts to comm.
 */
struct dense_part_t {
    int dense_id;
    int start_pos_on_dense;
    int length;

    int server_id;
    int start_pos_on_server;
    int part_id_on_server;
};

/**
 * @brief each server has an object of server_dense_part_t;
 * every server_dense_part_t contains server's all dense parts
 */
struct server_dense_part_t {
    std::vector<dense_part_t> parts;
    int part_num;
    int server_index;
    server_dense_part_t() : part_num(0), server_index(0) {}
};

class KeyParam;

/**
 * @brief class to store one model's parameters - dense and sparse data.
 */
class ModelParam {
public:
    /**
     * @brief constructor for model parameters.
    **/
    ModelParam(bool need_share_space);

    /**
     * @brief destructor for model parameters.
    **/
    ~ModelParam();

    /**
     * @brief get model size
     * @return int  : model size
    **/
    int dense_size() const {
        return _dense_size;
    }

    /**
     * @brief get model size
     * @return int  : model size
    **/
    int sparse_size() const {
        return _sparse_size;
    }

    int key_num() const {
        return _key_num;
    }

    /**
     * @brief get dense number
     * @return int : dense number
    **/
    int get_dense_num() const;

    /**
     * @brief get sparse number
     * @return int : sparse number
    **/
    int get_sparse_num() const;

    /**
     * @brief get const dense data (dense_t) by given dense index
     * @param [in] dense_index : int - dense index from 0
     * @return const dense_t*  : dense data of dense_index
    **/
    const dense_t* get_dense_data(int dense_index) const;

    /**
     * @brief get dense data (dense_t) by given dense index
     * @param [in] dense_index : int - dense index from 0
     * @return dense_t*  : dense data of dense_index
    **/
    dense_t* get_dense_data(int dense_index);

    /**
     * @brief get const sparse data (sparse_t) by given sparse index
     * @param [in] sparse_index : int - sparse index from 0
     * @return const sparse_t*  : sparse data of sparse_index
    **/
    const sparse_t* get_sparse_data(int sparse_index) const;

    /**
     * @brief get sparse data (sparse_t) by given sparse index
     * @param [in] sparse_index : int - sparse index from 0
     * @return sparse_t*  : sparse data of sparse_index
    **/
    sparse_t* get_sparse_data(int sparse_index);

    /**
     * @brief register one dense data
     * @param [in] row       : int - row size
     * @param [in] col       : int - column size
     * @param [in] psrc_data : real* - model parameter pointer
     * @return int           : 0-sucess; <0-error
    **/
    int register_dense_data(int row, int col, real* psrc_data, real* psrc_diff);

    /**
     * @brief register one sparse data
     * @param [in] vocabulary_size  : int - vocabulary size
     * @param [in] sparse_dim       : int - dimension for one word
     * @param [in] psrc_data : real* - model parameter pointer
     * @return int                  : 0-success; <0-error
    **/
    int register_sparse_data(int vocabulary_size, int sparse_dim, real* psrc_data, real* psrc_diff);

    /**
     * @brief register one dense data
     * @param [in] row       : int - row size
     * @param [in] col       : int - column size
     * @param [in] psrc_data : real* - model parameter pointer
     * @return int           : 0-sucess; <0-error
    **/
    int register_dense_data(int row, int col, real* psrc_data);

    /**
     * @brief register one sparse data
     * @param [in] vocabulary_size  : int - vocabulary size
     * @param [in] sparse_dim       : int - dimension for one word
     * @param [in] psrc_data : real* - model parameter pointer
     * @return int                  : 0-success; <0-error
    **/
    int register_sparse_data(int vocabulary_size, int sparse_dim, real* psrc_data);

    /**
     * @brief set key number of one sparse data without space allocation.
     * @param [in] sparse_index : int - sparse index
     * @param [in] key_num      : int - vocabulary size
     * @return int              : 0-success; <0-error
    **/
    int set_key_num(int sparse_index, int key_num);

    /**
     * @brief reset changed key number
     * @return void
    **/
    void reset_key_num();

    /**
     * @brief set changed key data.
     * @param [in] sparse_index : int - sparse index
     * @param [in] key_num      : int - changed key number
     * @param [in] pkey         : int*- wordid value
     * @return int              : 0-success; <0-error
    **/
    int set_changed_key(int sparse_index, int key_num, int* pkey);

    /**
     * @brief set all keys as changed key.
     * @return int : 0-success; <0-error
    **/
    int set_changed_key_all();

    /**
     * @brief insert one word id into set
     * @param [in] wordid : int - one word id
     * @return void
    **/
    int set_changed_key(const KeyParam& key_data);

    /**
     * @brief initialize this object with another object: allocate space and copy data.
     * @param [in] other : const ModelParam& - other model param
     * @return int       : 0-success; <0-error
    **/
    int copy_deep(const ModelParam& other);

    /**
     * @brief initialize this object with set data pointer to another object.
     * @param [in] other : const ModelParam& other - other model param
     * @return int       : 0-success; <0-error
    **/
    int copy_shallow(const ModelParam& other);

    /**
     * @brief copy value from another object.
     * @param [in] other : const ModelParam& other - other model param
     * @return int       : 0-success; <0-error
    **/
    int restore(const ModelParam& other);

    /**
     * @brief copy value from new to old.
     * @return int : 0-success; <0-error
    **/
    int restore_old();

    /**
     * @brief init space for pvalue old .
     * @return int : 0-success; <0-error
    **/
    int init_old();
private:
    /**
     * @brief constructor for model param
    **/
    ModelParam();

    /**
     * @brief copy constructor for model param
     * @param [in] other : const ModelParam& other - other model param
    **/
    ModelParam(const ModelParam& other);

    /**
     * @brief assgin operator for model param
     * @param [in] other : const ModelParam& other - other model param
     * @return ModelParam& : model parameter itself
    **/
    ModelParam& operator=(const ModelParam& other);

    /**
     * @brief set dense number
     * @param [in] num : int - dense number
     * @return int : 0-success; <0-error
    **/
    int set_dense_num(int num);

    /**
     * @brief set sparse number
     * @param [in] num : int - sparse number
     * @return int : 0-success; <0-error
    **/
    int set_sparse_num(int num);

    /**
     * @brief add one dense data: allocate space and copy data.
     * @param [in] row : int - row size
     * @param [in] col : int - column size
     * @param [in] psrc_data : int - model parameter pointer
     * @return int : 0-success; <0-error
    **/
    int add_dense_data(int row, int col, real* psrc_data);

    /**
     * @brief set one dense data pointer to psrc_data and just allocate space for keys.
     * @param [in] row : int - row size
     * @param [in] col : int - column size
     * @param [in] pvalue : real* - model parameter pointer
     * @return int : 0-success; <0-error
    **/
    int set_dense_data(int row, int col, real* pvalue, real* pvalue_diff);

    /**
     * @brief set one dense data pointer to psrc_data and just allocate space for keys.
     * @param [in] row : int - row size
     * @param [in] col : int - column size
     * @param [in] pvalue : real* - model parameter pointer
     * @return int : 0-success; <0-error
    **/
    int set_dense_data(int row, int col, real* pvalue);

    /**
     * @brief add one sparse data: allocate space and copy data.
     * @param [in] sparse_dim : int - dimension for one word embedding
     * @param [in] key_num    : int - vocabulary size
     * @param [in] pvalue     : real* - model prameter pointer
     * @return int : 0-success; <0-error
    **/
    int add_sparse_data(int sparse_dim, int key_num, real* pvalue);

    /**
     * @brief set one sparse data: allocate key space and set pvalue to pointer.
     * @param [in] sparse_dim : int - dimension for one word embedding
     * @param [in] key_num    : int - vocabulary size
     * @param [in] pvalue     : real* - model prameter pointer
     * @return int : 0-success; <0-error
    **/
    int set_sparse_data(int sparse_dim, int key_num, real* pvalue, real* pvalue_diff);

    /**
     * @brief set one sparse data: allocate key space and set pvalue to pointer.
     * @param [in] sparse_dim : int - dimension for one word embedding
     * @param [in] key_num    : int - vocabulary size
     * @param [in] pvalue     : real* - model prameter pointer
     * @return int : 0-success; <0-error
    **/
    int set_sparse_data(int sparse_dim, int key_num, real* pvalue);

    // model size
    int _dense_size;

    // model size
    int _sparse_size;

    // total key num
    int _key_num;

    // dense number
    int _dense_num;

    // dense data structure for each dense
    struct dense_t _dense[MAX_DENSE_NUM];

    // sparse number
    int _sparse_num;

    // sparse data structre for each sparse
    struct sparse_t _sparse[MAX_SPARSE_NUM];

    // share memory space for pvalue or not
    bool _share_space_pvalue;

    // share memory space for pvalue_old or not
    bool _share_space_pvalue_old;
};

/**
 * @brief class to store server's each sparse data's changed keys on workers.
**/
class KeyParam {
public:
    /**
     * @brief constructor for keyids recorder
    **/
    KeyParam();

    /**
     * @brief destructor for keyids recorder
    **/
    ~KeyParam();

    /**
     * @brief init key parameter for sparse data size from model parameter.
     * for each sparse data, there is a key array: 0-no change, 1-changed.
     * @param [in] model_param : const ModelParam& - model parameter
     * @return int : 0-success; <0-error
    **/
    int init(const ModelParam& model_param);

    /**
     * @brief get sprase number
     * @return int : sparse number
    **/
    int get_sparse_num() const;

    /**
     * @brief set one key as changed - key index is key id.
     * @param [in] sparse_index : int - sparse index
     * @param [in] key_index    : int - key index
     * @return int : 0-success; <0-error
    **/
    int set_key(int sparse_index, int key_index);

    /**
     * @brief get one sparse's all key data.
     * @param [in] sparse index : int - sparse index
     * @return const struct sparse_t* : key data
    **/
    const struct sparse_t* get_key(int sparse_index) const;

    /**
     * @brief clear data - reset all key to no change 0.
     * @return void
    **/
    void reset();

private:
    // sparse number
    int _sparse_num;

    // key data for multi-sparse
    sparse_t _sparse[MAX_SPARSE_NUM];
};

}

#endif  // MPI_MODEL_PARAM_H

/* vim: set ts=4 sw=4 sts=4 tw=100 */
