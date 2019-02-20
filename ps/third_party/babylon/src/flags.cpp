#include <gflags/gflags.h>

DEFINE_bool(babylon_pool_statistic, false, "enable alloc and hit statistic for debug");
DEFINE_uint64(babylon_pool_reserve, 128, "default reserve object num for each pool");
DEFINE_uint64(babylon_pool_cache_per_thread, 3, "default cache object num for each pthread");
DEFINE_uint64(babylon_pool_reserve_global, 64, "default reserve object num for global only, "
    "which are not cachable in pthread");
DEFINE_uint64(babylon_pool_create_thread_num, 1, "default thread_num to create objects, "
    "when init object_pool");

DEFINE_uint64(babylon_buffer_block_size, 10 << 20, "default block size for ReadOnlyBuffer");
