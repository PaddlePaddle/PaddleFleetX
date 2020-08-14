# ubuntu train
rm -rf data_small/ubuntu/train
mkdir -p data_small/ubuntu/train
head -n 1000 data/ubuntu/train.txt > data_small/ubuntu/train/train.txt
split -l 250 data_small/ubuntu/train/train.txt data_small/ubuntu/train/
rm -f data_small/ubuntu/train/train.txt
ls data_small/ubuntu/train/* > train.ubuntu.files

# ubuntu test
rm -rf data_small/ubuntu/test
mkdir -p data_small/ubuntu/test
head -n 100 data/ubuntu/test.txt > data_small/ubuntu/test/test.txt
ls data_small/ubuntu/test/* > test.ubuntu.files

# douban train
rm -rf data_small/douban/train
mkdir -p data_small/douban/train
head -n 1000 data/douban/train.txt > data_small/douban/train/train.txt
split -l 250 data_small/douban/train/train.txt data_small/douban/train/
rm -f data_small/douban/train/train.txt
ls data_small/douban/train/* > train.douban.files

# douban test
rm -rf data_small/douban/test
mkdir -p data_small/douban/test
head -n 100 data/douban/test.txt > data_small/douban/test/test.txt
ls data_small/douban/test/* > test.douban.files
