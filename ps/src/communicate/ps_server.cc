#include "glog/logging.h"
#include "table/table.h"
#include "communicate/ps_server.h"
#include "communicate/downpour_ps_server.h"

namespace paddle {
namespace ps {
REGISTER_CLASS(PSServer, DownpourBrpcPsServer);
REGISTER_CLASS(PsBaseService, DownpourPsService);

PSServer* PSServerFactory::create(const PSParameter& ps_config) {
    const auto& config = ps_config.server_param();
    //if (!config.has_downpour_server_param()) {
    if (!config.has_downpour_server_param()) {
         LOG(ERROR) << "miss downpour_server_param in ServerParameter";
         return NULL;
    }
    //if (!config.downpour_server_param().has_service_param()) {
    if (!config.downpour_server_param().has_service_param()) {
         LOG(ERROR) << "miss service_param in ServerParameter.downpour_server_param";
         return NULL;
    }
    //if (!config.downpour_server_param().service_param().has_server_class()) {
    if (!config.downpour_server_param().service_param().has_server_class()) {
         LOG(ERROR) << "miss server_class in ServerParameter.downpour_server_param.service_param";
         return NULL;
    }
    //const auto& service_param = config.downpour_server_param().service_param();
    const auto& service_param = config.downpour_server_param().service_param();
    PSServer* server = CREATE_CLASS(PSServer, service_param.server_class());
    if (server == NULL) {
        LOG(ERROR) << "server is not registered, server_name:" << service_param.server_class();
        return NULL;
    }
    TableManager::instance().initialize();
    /*
    if (server->configure(config) != 0 && server->initialize() != 0) {
        LOG(ERROR) << "server initialize failed, server_name:" << service_param.server_class();
        delete server;
        return NULL;
    }
    */
    return server;
}

}
}
