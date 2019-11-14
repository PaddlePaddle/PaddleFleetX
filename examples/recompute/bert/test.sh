PRETRAINED_CKPT_PATH=uncased_L-24_H-1024_A-16/params
DATA_PATH=xnli
bert_config_path=uncased_L-24_H-1024_A-16/bert_config.json
vocab_path=uncased_L-24_H-1024_A-16/vocab.txt
sh train_cls.sh $PRETRAINED_CKPT_PATH $bert_config_path $vocab_path $DATA_PATH
