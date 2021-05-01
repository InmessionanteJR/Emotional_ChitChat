gpuid=2

mark=$1
epoch_id=$2
pred_mark=$3

full_mark=${mark}_${epoch_id}_${pred_mark}

output=outputs/$mark


gt_file=data/data_v15_d1g10_transductive_for_base_ori/weibo_78wu_v15_d1g10f3_s10p3_0p35_0p25_1k_train_0_to_34_for_base.shuffled_h125ktrain_shuffled_v21_wo_train.test.ori.tgt
#gt_file=data/data_v15_d1g10_transductive_for_base_ori/wyl_test.tgt0

checkpoint=${output}/System_checkpoint${epoch_id}.pt
#checkpoint=${output}/System_checkpoint0.pt
condition_checkpoint=${output}/System_condition_generator_checkpoint${epoch_id}.pt

mkdir -p loggs results

function generate() {
CUDA_VISIBLE_DEVICES=${gpuid} \
python main.py \
    --run-mode test \
    --load-checkpoint $checkpoint \
    --load-condition_checkpoint $condition_checkpoint \
    --pred_output_file results/res$full_mark \
    > loggs/logg$full_mark 2> loggs/errg$full_mark
}

function evaluate() {
    python scripts/bleu.py $gt_file results/res$full_mark 0 > results/res${full_mark}.bleu
}

generate
evaluate
