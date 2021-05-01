gpuid=3
# sh generate_multi_epoches_and_test_sentiment_scores.sh RL75_tlsa top20 weibo_senti_100k
pretrained=$1 # eg: RL985
mark=$2 # eg: top20, beam5
Bert_pretrained=$3 # eg: weibo_senti_100k
Bert_mark=${pretrained}_${mark} # a personized string (NOTICE: must be '${pretrained}_${mark}', eg: RL985_top20)

output=outputs/$pretrained
gt_file=data/data_v15_d1g10_transductive_for_base_ori/weibo_78wu_v15_d1g10f3_s10p3_0p35_0p25_1k_train_0_to_34_for_base.shuffled_h125ktrain_shuffled_v21_wo_train.test.ori.tgt
mkdir -p results/${pretrained}_${mark}
mkdir -p loggs/${pretrained}_${mark}
mkdir -p loggs/${pretrained}_${mark}
mkdir -p Bert/logs/${Bert_mark}
mkdir -p Bert/checkpoints/${Bert_pretrained}/${Bert_mark}
mkdir -p logg/distinct_log/

function generate() {
CUDA_VISIBLE_DEVICES=${gpuid} \
python main.py \
    --run-mode test \
    --output-dir ${output} \
    --epoch-id ${epoch_id} \
    --pred-output-file results/${pretrained}_${mark}/res_${full_pretrained} \
    > loggs/${pretrained}_${mark}/logg_${full_pretrained} 2> loggs/${pretrained}_${mark}/errg_${full_pretrained}
}


function evaluate_bleu() {
    python scripts/bleu.py ${gt_file}    results/${pretrained}_${mark}/res_${full_pretrained} 0 > results/${pretrained}_${mark}/res_${full_pretrained}.bleu
}

function evaluate_dist() {
    input=results/${pretrained}_${mark}/res_${full_pretrained}
    output_dist=${input}.dist

    basename=$(basename $input)
    errlog=logg/distinct_log/${basename}.dist_log
    cat $input | python scripts/eval/distinct.py > $output_dist 2> $errlog
}

function test_sentiment() {
checkpoint_file=Bert/checkpoints/${Bert_pretrained}
CUDA_VISIBLE_DEVICES=${gpuid} \
python Bert/Bert_based_sentiment_classifier.py \
   --run-mode generate \
   --output-dir ${checkpoint_file}  \
   --output-name ${Bert_mark}_${epoch_id} \
   --texar-src-name ${Bert_mark}/res_${Bert_mark}_${epoch_id} \
   --Bert-mark ${Bert_mark} \
   > Bert/logs/${Bert_mark}/generate_log_${Bert_pretrained}_${Bert_mark}_${epoch_id} 2> Bert/logs/${Bert_mark}/generate_err_${Bert_pretrained}_${Bert_mark}_${epoch_id}
}



for epoch_id in {10..19}
do
     full_pretrained=${pretrained}_${mark}_${epoch_id}
     generate
     evaluate_bleu
     evaluate_dist
     test_sentiment
done