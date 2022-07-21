dir=data
depth=9
diff=100
fulldir=${dir}_${diff}_d${depth}

mkdir -p ${fulldir}

options="
uci
setoption name PruneAtShallowDepth value false
setoption name Use NNUE value true
setoption name Threads value 250
setoption name Hash value 10240
isready
generate_training_data depth 9 count 1000 random_multi_pv 4 random_multi_pv_diff 100 set_recommended_uci_options data_format binpack output_file_name ${fulldir}/gensfen.binpack book noob_3moves.epd seed ${RANDOM}${RANDOM}
quit"

echo "$options"
 
printf "$options" | ./stockfish > ${fulldir}/out.txt

echo "Done ${TID}"