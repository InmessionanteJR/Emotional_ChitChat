mark=$1
gpuid="0"
output=outputs/$mark


CUDA_VISIBLE_DEVICES=${gpuid} \
python main.py \
    --output-dir $output  \
    > logs/log_${mark} 2> logs/err_${mark}
