#!/bin/bash

FILES20="/archive3/group/chlgrp/twitter-collection-2020/twitter-2020-*.txt"
FILES21="/archive3/group/chlgrp/twitter-collection-2021/twitter-2021-*.txt"
FILES22="/archive3/group/chlgrp/twitter-collection-2022/twitter-2022-*.txt"

FILES=( $FILES20 $FILES21 $FILES22 )
total=${#FILES[@]}
echo Total number of files is ${total}
c=0


for f in "${FILES[@]}"
do
	if [ -e ${f} ]
	then
	    echo ${f} exists
	    stat -L -c "%a %G %U" ${f}
	    if [ ! -s ${HOME}/geo-twitter/datasets/world/${f:47:18}.txt ]
	    then
		echo filtered file is empty or does not exist
		sbatch collector.sh ${f}
		c=$((c+1))
	    else
		echo filtered dataset is already collected
	    fi
	else
	    echo ${f} does not exist
	fi
done


echo Total number of files is ${total}
echo Total number of scripts launched is ${c}
