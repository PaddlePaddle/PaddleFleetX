#include "table/downpour_dense_table.h"
#include "table/downpour_sparse_table.h"
#include "table/downpour_accessor.h"
#include "table/simple_copy_accessor.h"
#include "common/registerer.h"
#include "sgd/dense_sgd_factory.h"
#include "glog/logging.h"
#include <boost/preprocessor/repetition/repeat_from_to.hpp>
#include <boost/preprocessor/seq/elem.hpp>

DEFINE_bool(pslib_is_debug, false, "open debug mode in pslib");
DEFINE_bool(pslib_is_debug_push_sparse, false, "open debug mode in pslib");
DEFINE_int32(pslib_sparse_table_shard_num, 1000, "sparse table shard for save & load");
DEFINE_bool(pslib_is_update_grident_thread_save, true, "update_grident with thread-lock in pslib");

namespace paddle {
namespace ps {
    REGISTER_CLASS(Table, DownpourDenseTable);
    REGISTER_CLASS(Table, DownpourSparseTable);
    REGISTER_CLASS(ValueAccessor, DownpourDenseValueAccessor);
    REGISTER_CLASS(ValueAccessor, DownpourFeatureValueAccessor);
    REGISTER_CLASS(ValueAccessor, DownpourSparseValueAccessor);
    REGISTER_CLASS(ValueAccessor, SimpleCopyValueAccessor);

    int32_t TableManager::initialize() {
        static bool initialized = false;
        if (initialized) {
            return 0;
        }
        initialized = true;
        pslib_sgd_init();
        return 0;
    }

    int32_t Table::initialize(const TableParameter& config, 
        const FsClientParameter& fs_config) {
        _config = config;  
        if (initialize_accessor() != 0) {
            LOG(WARNING) << "Table accessor initialize failed";
            return -1;
        }
        if (_afs_client.initialize(fs_config) != 0) {
            LOG(WARNING) << "Table fs_client initialize failed"; 
            return -1;
        }
        return initialize();
    }

    int32_t Table::initialize_accessor() {
        if (!_config.has_accessor() || !_config.accessor().has_accessor_class()) {
            LOG(ERROR) << "missing accessor config in table, table_id:" << _config.table_id();
            return -1;
        }
        auto* accessor = CREATE_CLASS(ValueAccessor, _config.accessor().accessor_class())
        if (accessor == NULL) {
            LOG(ERROR) << "accessor is unregisteg, table_id:" << _config.table_id()
                << ", accessor_name:" << _config.accessor().accessor_class();
            return -1;
        }
        if (accessor->configure(_config.accessor()) || accessor->initialize() != 0) {
            LOG(ERROR) << " accessor initialize failed, table_id:" << _config.table_id()
                << ", accessor_name:" << _config.accessor().accessor_class();
            return -1;
        }
        _value_accesor.reset(accessor);
        return 0;
    }
    
}
}
