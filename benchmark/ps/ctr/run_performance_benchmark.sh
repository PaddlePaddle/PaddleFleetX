for thr in 40 ;
do 
    for bs in 32 64 128 256 512 1024 ;
    do
	python local_train.py --model_output_dir ${thr}_${bs}_models --num_passes 20 --thread_num $thr --batch_size $bs
    done
done
