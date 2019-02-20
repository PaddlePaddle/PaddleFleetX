#include "glog/logging.h"
#include "table/table.h"
#include "communicate/ps_client.h"
#include "communicate/downpour_ps_client.h"
#include "brpc/server.h"

namespace paddle {
namespace ps {

    REGISTER_CLASS(PSClient, DownpourBrpcPsClient);
    
    int32_t PSClient::configure(
        const PSParameter& config, PSEnvironment& env, size_t client_id) {
        _env = &env;
        _config = config;
        _client_id = client_id;
        //保证server与worker的 table配置一致
        _config.mutable_worker_param()->mutable_downpour_worker_param()->
            mutable_downpour_table_param()->CopyFrom(_config.server_param().
                                                     downpour_server_param().downpour_table_param());
        /*
        _config.mutable_worker_param()->mutable_downpour_worker_param()->
            mutable_downpour_table_param()->CopyFrom(_config.server_param().
            downpour_server_param().downpour_table_param());
        */
        //const auto& work_param = _config.worker_param().downpour_worker_param();
        const auto& work_param = _config.worker_param().downpour_worker_param();
        //for (size_t i = 0; i < work_param.downpour_table_param_size(); ++i) {
        for (size_t i = 0; i < work_param.downpour_table_param_size(); ++i) {
            auto* accessor = CREATE_CLASS(ValueAccessor, 
                                          work_param.downpour_table_param(i).accessor().accessor_class());
                                          //work_param.downpour_table_param(i).accessor().accessor_class());
            //accessor->configure(work_param.downpour_table_param(i).accessor());
            accessor->configure(work_param.downpour_table_param(i).accessor());
            accessor->initialize();
            //_table_accessors[work_param.downpour_table_param(i).table_id()].reset(accessor);
            _table_accessors[work_param.downpour_table_param(i).table_id()].reset(accessor);
        }
        return initialize();
    }

    PSClient* PSClientFactory::create(const PSParameter& ps_config) {
        const auto& config = ps_config.server_param();
        if (!config.has_downpour_server_param()) {
            LOG(ERROR) << "miss downpour_server_param in ServerParameter";
            return NULL;
        }
        /*
        if (!config.has_downpour_server_param()) {
             LOG(ERROR) << "miss downpour_server_param in ServerParameter";
            return NULL;
        }
        */
        if (!config.downpour_server_param().has_service_param()) {
            LOG(ERROR) << "miss service_param in ServerParameter.downpour_server_param";
            return NULL;
        }
        /*
        if (!config.downpour_server_param().has_service_param()) {
             LOG(ERROR) << "miss service_param in ServerParameter.downpour_server_param";
            return NULL;
        }
        */
        if (!config.downpour_server_param().service_param().has_client_class()) {
            LOG(ERROR) << "miss client_class in ServerParameter.downpour_server_param.service_param";
            return NULL;
        }
        /*
        if (!config.downpour_server_param().service_param().has_client_class()) {
             LOG(ERROR) << "miss client_class in ServerParameter.downpour_server_param.service_param";
            return NULL;
        }
        */
        const auto& service_param = config.downpour_server_param().service_param();
        //const auto& service_param = config.downpour_server_param().service_param();
        PSClient* client = CREATE_CLASS(PSClient, service_param.client_class());
        if (client == NULL) {
            LOG(ERROR) << "client is not registered, server_name:" << service_param.client_class();
            return NULL;
        }
        TableManager::instance().initialize();
        LOG(INFO) << "Create PSClient[" << service_param.client_class() << "] success";
        return client; 
    }

}
}
