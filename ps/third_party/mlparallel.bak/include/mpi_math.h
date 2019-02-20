/***************************************************************************
 *
 * Copyright (c) 2015 Baidu.com, Inc. All Rights Reserved
 * $Id$
 *
 **************************************************************************/

/**
* @file mpi_math.h
* @author wuzhihua02(wuzhihua02@baidu.com)
* @date 2015/1/15 15:50:00
* @version $Revision$
* @brief
*
**/
#ifndef MPIWRAPPER_MPI_MATH_H
#define MPIWRAPPER_MPI_MATH_H

#include <immintrin.h>
#include "mpi_common.h"

namespace mpi_wrapper {
#ifndef DTYPE_DOUBLE

#define __m256x __m256
#define __m128x __m128

static const unsigned int AVX_STEP_SIZE = 8;
static const unsigned int SSE_STEP_SIZE = 4;
static const unsigned int AVX_CUT_LEN_MASK = 7U;
static const unsigned int SSE_CUT_LEN_MASK = 3U;

#define _mm256_setzero_px _mm256_setzero_ps
#define _mm256_mul_px _mm256_mul_ps
#define _mm256_add_px _mm256_add_ps
#define _mm256_load_px _mm256_loadu_ps
#define _mm256_hadd_px _mm256_hadd_ps
#define _mm256_permute2f128_px _mm256_permute2f128_ps
#define _mm256_store_px _mm256_storeu_ps
#define _mm256_broadcast_sx _mm256_broadcast_ss
#define _mm256_castpx256_px128 _mm256_castps256_ps128
#define _mm256_max_px _mm256_max_ps
#define _mm256_sub_px _mm256_sub_ps
#define _mm256_set1_px _mm256_set1_ps
#define _mm256_sqrt_px _mm256_sqrt_ps
#define _mm256_div_px _mm256_div_ps
#define _mm_setzero_px _mm_setzero_ps
#define _mm_add_px _mm_add_ps
#define _mm_mul_px _mm_mul_ps
#define _mm_load_px _mm_loadu_ps
#define _mm_hadd_px _mm_hadd_ps
#define _mm_store_sx _mm_store_ss
#define _mm_store_px _mm_storeu_ps
#define _mm_load1_px _mm_load1_ps
#define _mm_max_px _mm_max_ps
#define _mm_sub_px _mm_sub_ps
#define _mm_set1_px _mm_set1_ps
#define _mm_sqrt_px _mm_sqrt_ps
#define _mm_div_px _mm_div_ps

#else

#define __m256x __m256d
#define __m128x __m128d

static const unsigned int AVX_STEP_SIZE = 4;
static const unsigned int SSE_STEP_SIZE = 2;
static const unsigned int AVX_CUT_LEN_MASK = 3U;
static const unsigned int SSE_CUT_LEN_MASK = 1U;

#define _mm256_setzero_px _mm256_setzero_pd
#define _mm256_mul_px _mm256_mul_pd
#define _mm256_add_px _mm256_add_pd
#define _mm256_load_px _mm256_loadu_pd
#define _mm256_hadd_px _mm256_hadd_pd
#define _mm256_permute2f128_px _mm256_permute2f128_pd
#define _mm256_store_px _mm256_storeu_pd
#define _mm256_broadcast_sx _mm256_broadcast_sd
#define _mm256_castpx256_px128 _mm256_castpd256_pd128
#define _mm256_max_px _mm256_max_pd
#define _mm256_sub_px _mm256_sub_pd
#define _mm256_set1_px _mm256_set1_pd
#define _mm256_sqrt_px _mm256_sqrt_pd
#define _mm256_div_px _mm256_div_pd
#define _mm_setzero_px _mm_setzero_pd
#define _mm_add_px _mm_add_pd
#define _mm_mul_px _mm_mul_pd
#define _mm_load_px _mm_loadu_pd
#define _mm_hadd_px _mm_hadd_pd
#define _mm_store_sx _mm_store_sd
#define _mm_store_px _mm_store_pd
#define _mm_load1_px _mm_load1_pd
#define _mm_max_px _mm_max_pd
#define _mm_sub_px _mm_sub_pd
#define _mm_set1_px _mm_set1_pd
#define _mm_sqrt_px _mm_sqrt_pd
#define _mm_div_px _mm_div_pd
#endif

/**
 * @brief z += x + y
 * @param [in] input vector : const real* - x
 * @param [in] input vecotr : const real* - y
 * @param [out] out vector  : real* - z
 * @param [in] len          : size_t - len of vector in real
 * @return void
 **/
inline void sse_sub(const real* x, const real* y, real* z, size_t len) {
    unsigned int jjj, lll;
    jjj = lll = 0;

#if defined(USE_AVX)
    lll = len&~AVX_CUT_LEN_MASK;

    for (jjj = 0; jjj < lll; jjj += AVX_STEP_SIZE) {
        _mm256_store_px(z + jjj,
                        _mm256_add_px(_mm256_load_px(z + jjj),
                                      _mm256_sub_px(_mm256_load_px(x + jjj),
                                              _mm256_load_px(y + jjj))));
    }

    //ML_DEBUG("avx_sub\n");
#elif defined(USE_SSE)
    lll = len&~SSE_CUT_LEN_MASK;

    for (jjj = 0; jjj < lll; jjj += SSE_STEP_SIZE) {
        _mm_store_px(z + jjj,
                     _mm_add_px(_mm_load_px(z + jjj),
                                _mm_sub_px(_mm_load_px(x + jjj),
                                           _mm_load_px(y + jjj))));
    }

    //ML_DEBUG("sse_sub\n");
#endif

    for (; jjj < len; jjj++) {
        z[jjj] += x[jjj] - y[jjj];
    }
}

/**
 * @brief z = x - y
 * @param [in] input vector : const real* - x
 * @param [in] input vecotr : const real* - y
 * @param [out] out vector  : real* - z
 * @param [in] len          : size_t - len of vector in real
 * @return void
 **/
inline void sse_sub_override(const real* x, real* y, real* z, size_t len) {
    unsigned int jjj, lll;
    jjj = lll = 0;

#if defined(USE_AVX)
    lll = len&~AVX_CUT_LEN_MASK;

    for (jjj = 0; jjj < lll; jjj += AVX_STEP_SIZE) {
        _mm256_store_px(z + jjj,
                        _mm256_sub_px(_mm256_load_px(x + jjj),
                                      _mm256_load_px(y + jjj)));
    }

#elif defined(USE_SSE)
    lll = len&~SSE_CUT_LEN_MASK;

    for (jjj = 0; jjj < lll; jjj += SSE_STEP_SIZE) {
        _mm_store_px(z + jjj,
                     _mm_sub_px(_mm_load_px(x + jjj),
                                _mm_load_px(y + jjj)));
    }

#endif

    for (; jjj < len; jjj++) {
        z[jjj] = x[jjj] - y[jjj];
    }
}

/**
 * @brief y += x * alpha
 * @param [in] input vector : const real* & - x
 * @param [in] alpha       : real -  alpha
 * @param [out] out vector : real* - y
 * @param [in] len         : size_t - len of vector in real
 * @return void
 **/
// y += x * alpha
inline void sse_add(const real* x, real alpha, real* y, size_t len) {
    unsigned int jjj, lll;
    jjj = lll = 0;

#if defined(USE_AVX)
    lll = len&~AVX_CUT_LEN_MASK;
    __m256x mm_alpha = _mm256_broadcast_sx(&alpha);

    for (jjj = 0; jjj < lll; jjj += AVX_STEP_SIZE) {
        _mm256_store_px(y + jjj,
                        _mm256_add_px(_mm256_load_px(y + jjj),
                                      _mm256_mul_px(_mm256_load_px(x + jjj),
                                              mm_alpha)));
    }

#elif defined(USE_SSE)
    lll = len&~SSE_CUT_LEN_MASK;
    __m128x mm_alpha = _mm_load1_px(&alpha);

    for (jjj = 0; jjj < lll; jjj += SSE_STEP_SIZE) {
        _mm_store_px(y + jjj,
                     _mm_add_px(_mm_load_px(y + jjj),
                                _mm_mul_px(_mm_load_px(x + jjj),
                                           mm_alpha)));
    }

#endif

    for (; jjj < len; jjj++) {
        y[jjj] += x[jjj] * alpha;
    }
}

/**
 * @brief y += x
 * @param [in] input vector : const real* & - x
 * @param [out] out vector : real* - y
 * @param [in] len         : size_t - len of vector in real
 * @return void
 **/
// y += x
inline void sse_add(const real* x, real* y, size_t len) {
    unsigned int jjj, lll;
    jjj = lll = 0;

#if defined(USE_AVX)
    lll = len&~AVX_CUT_LEN_MASK;

    for (jjj = 0; jjj < lll; jjj += AVX_STEP_SIZE) {
        _mm256_store_px(y + jjj,
                        _mm256_add_px(_mm256_load_px(y + jjj),
                                      _mm256_load_px(x + jjj)));
    }

#elif defined(USE_SSE)
    lll = len&~SSE_CUT_LEN_MASK;

    for (jjj = 0; jjj < lll; jjj += SSE_STEP_SIZE) {
        _mm_store_px(y + jjj,
                     _mm_add_px(_mm_load_px(y + jjj),
                                _mm_load_px(x + jjj)));
    }

#endif

    for (; jjj < len; jjj++) {
        y[jjj] += x[jjj];
    }
}
}
#endif  // MPI_MATH_H

/* vim: set ts=4 sw=4 sts=4 tw=100 */
