#input=ft_local/all_78w_user_v0_129m.ori
#
#output_dict=weibo_78w_v00.2wdict
#output_src_tgt=weibo_78w_v00.src_tgt
#output_src=weibo_78w_v00.src
#output_tgt=weibo_78w_v00.tgt


input=train400w

output_dict=dict400w
output_src_tgt=train400w.src_tgt
output_src=train400w.src
output_tgt=train400w.tgt


mkdir -p log
function tokenize() {
# tokenize (ch-level segment over original data)

#cat $input | python2 tokenize.py > $output_src_tgt 2> log/log1
cat $input | python2 tokenize.py > $output_src_tgt 2> log/log1
}

# build dict
function build_dict() {
sh build_dict.sh $output_src_tgt $output_dict
}

# filter
function filter_by_dict() {
cat $output_src_tgt | python filter_by_dict.py $output_dict $output_src $output_tgt
}

# term2id
function term2id() {
cat $output_src_tgt | python2 term2id.py $output_dict".with_id" $output_src $output_tgt
}

tokenize
#build_dict
filter_by_dict
#term2id
