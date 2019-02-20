/***************************************************************************
 *
 * Copyright (c) 2015 Baidu.com, Inc. All Rights Reserved
 * $Id$
 *
 **************************************************************************/

/**
* @file mpi_timer.h
* @author wuzhihua02(wuzhihua02@baidu.com)
* @date 2015/1/15 15:50:00
* @version $Revision$
* @brief
*
**/
#ifndef MPIWRAPPER_MPI_TIMER_H
#define MPIWRAPPER_MPI_TIMER_H

#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include "mpi_common.h"
namespace mpi_wrapper {
/**
 * @brief class to record time start and end
 */
class Timer {
public:
    /**
     * @brief constructor for timer.
    **/
    Timer();

    /**
     * @brief start for timer.
    **/
    void start();

    /**
     * @brief end for timer.
    **/
    void end();

    /**
     * @brief get end time.
     * @return struct timeval : end time
    **/
    struct timeval get_endtime();

    /**
     * @brief get time span.
     * @return long : time span
    **/
    long elapsed() const;

    /**
     * @brief get time span in usecond.
     * @return double : time span in usecond
    **/
    double usec_elapsed() const;

    /**
     * @brief get time span in msecond.
     * @return double : time span in msecond
    **/
    double msec_elapsed() const;

    /**
     * @brief get time span in second.
     * @return double : time span in second
    **/
    double sec_elapsed() const;

private:
    // start time
    struct timeval _start;

    // end time
    struct timeval _end;
};

/**
 * @brief get current time in format "%d-%d-%d-%d-%d-%d"
 * @return const char*: current time str
**/
const char* get_current_time();

/**
 * @brief get current time in format "%d-%d-%d,%d:%d"
 * @return const char* : current time str
**/
const char* get_current_time2();

/**
 * @brief get minutes span between last and now.
 * @return double : time span in minutes
**/
double get_diff_minute_fromnow(struct timeval last);
}
#endif  // MPI_TIMER_H

/* vim: set ts=4 sw=4 sts=4 tw=100 */
