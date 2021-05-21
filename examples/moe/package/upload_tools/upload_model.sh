#!/bin/bash
# Copyright (c) 2016 Baidu, Inc. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#set -x

source ./init_env.sh
source ../hadoop_functions.sh
source ../common_function.sh


function do_upload_model()
{
    local filepath=$1
    log_info "filepath=${filepath}"
    real_rank=`printf "%05d" ${PADDLE_TRAINER_ID}`
    log_info "real_rank: ${real_rank}"
    log_info "hadoop_put_file ${filepath}"

    dest_remote_path=${COMBINED_OUTPUT_PATH}/rank-${real_rank}/`basename ${filepath}`
    hadoop_test ${HADOOP_HOME} ${FS_NAME} ${FS_UGI} \
                 ${dest_remote_path} \
                ${DFS_USE_NATIVE_API} ${DFS_AGENT_PORT}

    if [[ $? -eq 0 ]]; then
        log_info "[${dest_remote_path}] has been exist , will delete firstly"
        hadoop_rmr ${HADOOP_HOME} ${FS_NAME} ${FS_UGI} \
                    ${dest_remote_path} \
                    ${DFS_USE_NATIVE_API} ${DFS_AGENT_PORT}
        if [[ $? -ne 0 ]]; then
            log_error "hadoop_rmr ${dest_remote_path} failed"
        fi
    fi
    log_info "will upload [${filepath}] to [${COMBINED_OUTPUT_PATH}/rank-${real_rank}/]"
    hadoop_put_file ${HADOOP_HOME} ${FS_NAME} ${FS_UGI} \
                    ${filepath} ${COMBINED_OUTPUT_PATH}/rank-${real_rank}/ \
                    ${DFS_USE_NATIVE_API} ${DFS_AGENT_PORT}
}

function detect_newpass_and_upload()
{
    save_dir="./${SYS_TMP_MULTIFILE_DIR}/output/"
    for filename in `ls -t ${save_dir}`; do
       log_info "start to upload model: ${save_dir}/${filename}"
       #start to upload them to hdfs
       do_upload_model "${save_dir}/${filename}"
       #5 second
       sleep 5
    done
}

function main(){

    # Will Get [ENV] PADDLE_TRAINER_ID from file of train_id
    if [[ -z ${PADDLE_TRAINER_ID} ]]; then
        get_trainer_id
    fi

    real_rank=`printf "%05d" ${PADDLE_TRAINER_ID}`
    log_info "upload_model real_rank: ${real_rank}"
    hadoop_test ${HADOOP_HOME} ${FS_NAME} ${FS_UGI} \
                ${COMBINED_OUTPUT_PATH}/rank-${real_rank}/ \
                ${DFS_USE_NATIVE_API} ${DFS_AGENT_PORT}
    if [[ $? -ne 0 ]]; then
        log_info "will create remote dir: ${COMBINED_OUTPUT_PATH}/rank-${real_rank}/"
        hadoop_mkdir ${HADOOP_HOME} ${FS_NAME} ${FS_UGI} \
                     ${COMBINED_OUTPUT_PATH}/rank-${real_rank}/ \
                     ${DFS_USE_NATIVE_API} ${DFS_AGENT_PORT}
    fi

    pushd ${TRAIN_WORKSPACE}
        log_info "PADDLE_TRAINER_ID=${PADDLE_TRAINER_ID}"
        detect_newpass_and_upload
    popd
}

main
