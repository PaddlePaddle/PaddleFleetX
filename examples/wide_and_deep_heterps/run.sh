mkdir -p ./log && hostname
# download build environment
sh ./build_env.sh &> ./log/build_env.log

# export environment config
export GLOG_v=1
source ./heterps.bashrc

# run server 
# port must be the same in run_psgpu.sh
sh run_psgpu.sh PSERVER 8500 0 &

# run worker
sh run_psgpu.sh TRAINER 8200 0 &
