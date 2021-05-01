input=$1 #d3
output=$2 #weibo.2w.dict
tmp=${output}".tmp"
ori=${output}".ori"
output_with_id=${output}".with_id"
vocab_size=20000

cat $input | awk '{for(i=1;i<=NF;i++){if(!($i in d)){d[$i]=1} else{d[$i]+=1}}} END{for(token in d){print token "\t" d[token]}}' > $tmp 
cat $tmp | sort -n -r -k 2,2 > $ori
head -$vocab_size $ori | awk -F "\t" '{print $1}' > $output
cat $output | awk '{print $1 "\t" NR-1}' > ${output_with_id}
