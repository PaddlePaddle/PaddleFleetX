#pragma once
#include <gflags/gflags.h>
#include <vector>
#include <memory>
#include <string>
#include <stdio.h>
#include <unordered_set>
#include <netinet/in.h>
#include <arpa/inet.h> 
#include <algorithm>
#include <glog/logging.h>

DECLARE_bool(pslib_open_strict_check);
DECLARE_bool(enable_dense_gradient_compress);
DECLARE_bool(pslib_enable_pull_dense_compress);

namespace paddle {
namespace ps {

struct PSHost {
    std::string ip;
    uint32_t port;
    uint32_t rank;
    //|---ip---|---port---|--rank--|
    //|-32bit--|--20bit---|--12bit-|
    uint64_t serialize_to_uint64() {
        uint64_t host_label = 0;
        host_label = inet_addr(ip.c_str());
        host_label = host_label << 32;
        host_label += (port << 12);
        host_label += rank;
        return host_label;
    }
    void parse_from_uint64(uint64_t host_label) {
        static uint64_t rank_label_mask = (1L << 12) - 1;
        static uint64_t port_label_mask = (1L << 20) - 1;
        rank = host_label & rank_label_mask;
        port = (host_label >> 12) & port_label_mask;
        uint32_t ip_addr = (host_label >> 32);
        ip =  inet_ntoa(*(in_addr*)&ip_addr);
    }
};

class PSEnvironment {
    public:
        explicit PSEnvironment() {};
        virtual ~PSEnvironment() {};
        //收集server列表
        virtual int32_t gather_ps_servers() {
            return gather_ps_hosts(_ps_server_list, _ps_server_sign_set);
        }
        virtual int32_t set_ps_servers(uint64_t* host_sign_list, int node_num) {
            return 0;
        }
        virtual uint64_t get_local_host_sign() {return 0;}
        //获取ps_server列表
        virtual std::vector<PSHost> get_ps_servers() const {
            return _ps_server_list;
        }
        //向env注册加入ps_server
        virtual int32_t registe_ps_server(const std::string& ip, uint32_t port, int32_t rank) {
            return registe_ps_host(ip, port, rank, _ps_server_list, _ps_server_sign_set);
        }

        //收集client列表
        virtual int32_t gather_ps_clients() {
            return gather_ps_hosts(_ps_client_list, _ps_client_sign_set);
        }
        //获取ps_client列表
        virtual std::vector<PSHost> get_ps_clients() const {
            return _ps_client_list;
        }
        //向env注册加入ps_server
        virtual int32_t registe_ps_client(const std::string& ip, uint32_t port, int32_t rank) {
            return registe_ps_host(ip, port, rank, _ps_client_list, _ps_client_sign_set);
        }
    protected:
        //收集host列表
        virtual int32_t gather_ps_hosts(std::vector<PSHost>& host_list, std::unordered_set<uint64_t>& sign_set) = 0;
        //注册一个host
        virtual int32_t registe_ps_host(const std::string& ip, uint32_t port, int32_t rank, 
            std::vector<PSHost>& host_list, std::unordered_set<uint64_t>& sign_set) {
            PSHost host;
            host.ip = ip;
            host.port = port;
            host.rank = rank;
            if (sign_set.count(host.serialize_to_uint64()) > 0) {
                LOG(WARNING) << "ps-host :" << host.ip << ":" << host.port 
                    << ", rank:" << host.rank << " already register, ignore register";
            } else {
                host_list.push_back(host);
                sign_set.insert(host.serialize_to_uint64());
            }
            return 0;
        }

        std::vector<PSHost> _ps_client_list;
        std::unordered_set<uint64_t> _ps_client_sign_set; //for unique filter

        std::vector<PSHost> _ps_server_list;
        std::unordered_set<uint64_t> _ps_server_sign_set; //for unique filter
};

class MpiPSEnvironment : public PSEnvironment {
    public:
        explicit MpiPSEnvironment() {};
        virtual ~MpiPSEnvironment() {};
    protected:
        virtual int32_t gather_ps_hosts(
            std::vector<PSHost>& host_list, std::unordered_set<uint64_t>& sign_set) override;
};

class PaddlePSEnvironment : public PSEnvironment {
    public:
        explicit PaddlePSEnvironment() {};
        virtual ~PaddlePSEnvironment() {};

        virtual int32_t set_ps_servers(uint64_t* host_sign_list, int node_num) {
            _ps_server_list.clear();
            _ps_server_sign_set.clear();
            for (size_t i = 0; i < node_num; ++i) {
                if (host_sign_list[i] > 0) {
                    PSHost host;
                    host.parse_from_uint64(host_sign_list[i]);
                    _ps_server_list.push_back(host);
                    _ps_server_sign_set.insert(host.serialize_to_uint64());
                }
            }
            std::sort(_ps_server_list.begin(), _ps_server_list.end(), 
                [](const PSHost& h1, const PSHost& h2) {
                    return h1.rank < h2.rank;
            });
            return 0;
        }

        virtual uint64_t get_local_host_sign() {
            if (_ps_client_list.size() > 0) {
                return _ps_client_list[0].serialize_to_uint64();
            } else {
                return 0;
            }            
        }

        virtual int32_t gather_ps_clients() {
            
            for (int i = 0; i < _ps_server_list.size(); ++i) {
                _ps_client_list.push_back(_ps_server_list[i]);
            }
            for (std::unordered_set<uint64_t>::iterator i = _ps_server_sign_set.begin(); i != _ps_server_sign_set.end(); i++)  {
                _ps_client_sign_set.insert(*i);
            }
            return 0;
        }
        virtual int32_t gather_ps_hosts(
            std::vector<PSHost>& host_list, std::unordered_set<uint64_t>& sign_set) override {
                /*
                std::sort(host_list.begin(), host_list.end(), 
                    [](const PSHost& h1, const PSHost& h2) {
                        return h1.rank < h2.rank;
                });
                std::stringstream list_str;
                for (auto& ps_host : host_list) {
                    list_str << ps_host.serialize_to_uint64()<< ",";
                }
                LOG(INFO) << "PsHost: " << list_str.str();
                */
                LOG(ERROR) << "[error] don't use this in paddle ps environment!";
                return 0;
            }
};

class LocalPSEnvironment : public PSEnvironment {
    public:
        explicit LocalPSEnvironment() {};
        virtual ~LocalPSEnvironment() {};
    protected:
        virtual int32_t gather_ps_hosts(
            std::vector<PSHost>& host_list, std::unordered_set<uint64_t>& sign_set) override {
            std::sort(host_list.begin(), host_list.end(), 
                [](const PSHost& h1, const PSHost& h2) {
                    return h1.rank < h2.rank;
            });
            std::stringstream list_str;
            for (auto& ps_host : host_list) {
                list_str << ps_host.serialize_to_uint64()<< ",";
            }
            LOG(INFO) << "PsHost: " << list_str.str();
            return 0;
        }
};

}
}
