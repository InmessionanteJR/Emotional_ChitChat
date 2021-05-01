mark=$1
gpuid=4

output=outputs/$mark

#bigstore_path=../../bigstore_socialNetConv/
#config_data=configs/config_data_debug_dv01_d3_smalldata.py
#config_data=config_data.py
#mkdir -p $bigstore_path"/"$output
mkdir -p logs
CUDA_VISIBLE_DEVICES=${gpuid} \
python main.py \
    --output-dir $output  \
    > logs/log$mark 2> logs/err$mark