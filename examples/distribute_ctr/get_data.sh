wget --no-check-certificate https://fleet.bj.bcebos.com/ctr_data.tar.gz
tar -zxvf ctr_data.tar.gz
mv ./raw_data ./train_data
echo "Complete data download."
echo "Train data stored in ./train_data folder."
echo "Test data stored in ./test_data folder."