device=$1
ckpt_dir=$2

CUDA_VISIBLE_DEVICES=$device python tools/eval.py -c ppfleetx/configs/nlp/gpt/eval_gpt_345M_single_card.yaml -o Engine.save_load.ckpt_dir=$ckpt_dir  -o Offline_Eval.cloze_eval=True -o Offline_Eval.batch_size=16  
