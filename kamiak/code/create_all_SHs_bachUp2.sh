#!/bin/bash
cd /home/a.norouzikandelati/Projects/Tillage_mapping/

batch_size=1000
batch_number=1
while [ $batch_number -le 100 ]
do
  cp template.sh    ./qsubs/q_Jeol_$batch_number.sh
  sed -i s/batch_size/"$batch_size"/g    ./qsubs/q_Jeol_$batch_number.sh
  sed -i s/batch_number/"$batch_number"/g  ./qsubs/q_Jeol_$batch_number.sh
  let "batch_number+=1" 
done