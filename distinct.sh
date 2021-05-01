mark=$1
input=results/$mark
output=${input}.dist

basename=$(basename $input)
errlog=logg/distinct_log/${basename}.dist_log

mkdir -p logg/distinct_log/
#cut -f 2 $input | python scripts/eval/distinct.py > $output 2> $errlog
cat $input | python scripts/eval/distinct.py > $output 2> $errlog
