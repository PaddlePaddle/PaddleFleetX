#ifdef MPI_TRAIN_ENV
//#include <mpi.h>
#endif
#include "communicate/ps_env.h"

namespace paddle {
namespace ps {


int32_t MpiPSEnvironment::gather_ps_hosts(
    std::vector<PSHost>& host_list, std::unordered_set<uint64_t>& sign_set) {
#ifdef MPI_TRAIN_ENV 
            /*
            uint64_t local_host_sign = 0;
            if (host_list.size() > 0) {
                local_host_sign = host_list[0].serialize_to_uint64();
            }
            int mpi_node_num = 0;
            int mpi_node_rank = 0;
            MPI_Comm_size(MPI_COMM_WORLD, &mpi_node_num);
            MPI_Comm_rank(MPI_COMM_WORLD, &mpi_node_rank);
            MPI_Barrier(MPI_COMM_WORLD);         
            uint64_t host_sign_list[mpi_node_num];
            host_sign_list[mpi_node_rank] = local_host_sign;
            MPI_Allgather(MPI_IN_PLACE, 0, MPI_UNSIGNED_LONG,
                host_sign_list, 1, MPI_UNSIGNED_LONG, MPI_COMM_WORLD);
            host_list.clear();
            sign_set.clear();
            for (size_t i = 0; i < mpi_node_num; ++i) {
                if (host_sign_list[i] > 0) {
                    PSHost host;
                    host.parse_from_uint64(host_sign_list[i]);
                    host_list.push_back(host);
                    sign_set.insert(host.serialize_to_uint64());
                }
            }
            std::sort(host_list.begin(), host_list.end(), 
                [](const PSHost& h1, const PSHost& h2) {
                    return h1.rank < h2.rank;
            });
            */
#else
            LOG(ERROR) << "Not build in mpi env!! must compile with -DMPI_TRAIN_ENV";
#endif
            return 0;
        }
}
}
