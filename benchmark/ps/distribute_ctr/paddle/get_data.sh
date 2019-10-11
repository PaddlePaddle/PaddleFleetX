#!/bin/bash
wget --no-check-certificate https://fleet.bj.bcebos.com/ctr_data.tar.gz
tar -zxvf ctr_data.tar.gz

mv raw_data train_data
