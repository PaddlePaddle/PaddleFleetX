/***************************************************************************
 *
 * Copyright (c) 2015 Baidu.com, Inc. All Rights Reserved
 * $Id$
 *
 **************************************************************************/

/**
* @file mpi_acevaluation.h
* @author wuzhihua02(wuzhihua02@baidu.com)
* @date 2015/1/15 15:50:00
* @version $Revision$
* @brief
*
**/
#ifndef MPIWRAPPER_MPI_ACEVALUATOR_H
#define MPIWRAPPER_MPI_ACEVALUATOR_H

#include <math.h>
#include <stdlib.h>
#include <sys/time.h>
#include <map>
#include <string>
#include <sstream>
#include "mpi_common.h"
#include "mpi_util.h"
namespace mpi_wrapper {

/**
 * @brief class for ac/dx/click positive-negative order score calculation.
 */
class AcEvaluator {
public:
    /**
     * @brief constructor for ac/dx/click score calculation.
    **/
    AcEvaluator();

    /**
     * @brief calculate score for qid files
     * @param [in] qid_file   : qid file
     * @param [in] score_file : file for output score
     * @param [in] model_name : model name
     * @param [out] out_score : score result
     * @return int - 0: success; -1: fail
    **/
    int calculate_score(const char* qid_file,
                        const char* score_file,
                        const char* model_name,
                        float& out_score);
private:
    void f();

    static int _s_scale[5];
    float _score[MAX_URLS_ONE_QUERY];
    int _label[MAX_URLS_ONE_QUERY];
    int _level_num[128];
    float _level_sum[128];
    float _level_squared_sum[128];

    int _right;
    int _wrong;
    int _equal;
    int _r[128];
    std::map<std::string, int> _r_d;
    int _w[128];
    std::map<std::string, int> _w_d;
    int _e[128];
    std::map<std::string, int> _e_d;
    int _un;
    std::string _preq;
    int _qnum;
    std::vector<std::string> _tokens;
    int _nr;
    double _z;
    double _z2;
    double _y;
    double _y2;
    double _zy;
    int _n;
    int _max_level;
};

}
#endif  // MPI_ACEVALUATOR_H

/* vim: set ts=4 sw=4 sts=4 tw=100 */
