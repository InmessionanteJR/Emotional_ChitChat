input=ft_local/xaa
#input=xaah100
output_dict=weibo_78w_v00_d1.2wdict
output_src_tgt=weibo_78w_v00_d0.src_tgt
output_src=weibo_78w_v00_d1.src
output_tgt=weibo_78w_v00_d1.tgt

mkdir -p log
function tokenize() {
# tokenize (ch-level segment over original data)
#cat $input | python2 tokenize.py $output_src $output_tgt > log/log1 2> log/err1
cat $input | python2 tokenize.py > $output_src_tgt 2> log/log1 
}

# build dict
function build_dict() {
#cat $output_src $output_tgt > $output_src".merge_with_tgt_for_dict"
#sh build_dict.sh $output_src".merge_with_tgt_for_dict" $output_dict
#rm -f $output_src".merge_with_tgt_for_dict"
sh build_dict.sh $output_src_tgt $output_dict
}

# filter
function filter_by_dict() {
#awk -F "\t" 'NR==FNR{d[$i]=1} NR!=FNR{for(i=1;i<NF;i++){if($i in d){printf("%s ", $i)} if($NF in d){print $NF}else{print ""}}}' | python rstrip.py > 
cat $output_src_tgt | python filter_by_dict.py $output_dict $output_src $output_tgt
}

# term2id
function term2id() {
cat $output_src_tgt | python term2id.py $output_dict".with_id" $output_src $output_tgt
}

#tokenize
#build_dict
#filter_by_dict
term2id
