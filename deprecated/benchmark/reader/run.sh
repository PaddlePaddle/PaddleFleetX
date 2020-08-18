echo "Begin to run reader benchmarks"

rm result*.txt

for num_threads in 1 2 4 8 16 24 32 38
do
    LIST="pil cv2 visreader libjpeg base64"
    for type in $LIST
    do
        echo "type: $type, num_threads: $num_threads" > result_${type}_${num_threads}.txt
        if [ $type == "visreader" ]
        then
            ./python/bin/python -u test.py $type --data_dir=/ssd2/lilong/ImageNet/output --num_threads=$num_threads >> result_${type}_${num_threads}.txt
        else
            ./python/bin/python -u test.py $type --data_dir=/ssd2/lilong/ImageNet --num_threads=$num_threads >> result_${type}_${num_threads}.txt
        fi
    done
done
