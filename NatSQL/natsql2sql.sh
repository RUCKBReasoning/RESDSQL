#!/bin/bash

dev_or_train=$1
natsql_or_natsqlg=$2

if [ "$dev_or_train" = "dev" ];
then
Path="NatSQLv1_6/dev.json"
elif [ "$dev_or_train" = "train" ];
then
Path="NatSQLv1_6/train_spider.json"
else
echo "The first paprmeter must be dev or train."
exit
fi


if [ "$natsql_or_natsqlg" = "natsql" ];
then
python run.py --natsql_file $Path --remove_groupby_from_natsql 1>results.sql
elif [ "$natsql_or_natsqlg" = "natsqlg" ];
then
python run.py --natsql_file $Path  
else
echo "The second paprmeter must be natsql or natsqlg."
exit
fi
