mark=$1
gpuid="2"

checkpoint_file=checkpoints/$mark


CUDA_VISIBLE_DEVICES=${gpuid} \
python Bert_based_sentiment_classifier.py \
    --output-dir $checkpoint_file  \
    > logs/train_log_$mark 2> logs/train_err_$mark
