## download raw data
wget --no-check-certificate https://fleet.bj.bcebos.com/ctr_data.tar.gz
tar -zxvf ctr_data.tar.gz
mv raw_data train_data_raw

# begin to process raw data
mkdir train_data_processed
python data_dnn_process.py

# delete raw data
mv train_data_processed train_data
rm -rf ./train_data_raw
