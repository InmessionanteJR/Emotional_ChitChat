mark=$1
gpuid="1"

checkpoint_file=checkpoints/$mark


CUDA_VISIBLE_DEVICES=${gpuid} \
python Bert_based_sentiment_classifier.py \
    --run-mode test \
    --output-dir $checkpoint_file  \
    > logs/test_log_$mark 2> logs/test_err_$mark