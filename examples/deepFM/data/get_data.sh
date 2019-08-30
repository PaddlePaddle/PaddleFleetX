#!/bin/bash

wget --no-check-certificate https://fleet.bj.bcebos.com/criteo_after_preprocessing.tar.gz
tar zxvf criteo_after_preprocessing.tar.gz
cp ./aid_data/feat_dict_10.pkl2 ../
