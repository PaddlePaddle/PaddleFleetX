wget --no-check-certificate https://fleet.bj.bcebos.com/simnet_bow_data.tar.gz
tar -zxvf simnet_bow_data.tar.gz
mv train_raw train_data
mkdir test_data
cd train_data
mv part-00099.raw part-00098.raw part-00097.raw part-00096.raw -t ../test_data/
echo "Data download complete"
