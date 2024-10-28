#!/bin/bash
cd /home/a.norouzikandelati/Projects/Tillage_mapping/codes/

num_batches=1000
batch_number=1
while [ $batch_number -le $num_batches ]
do
  cp template.sh    ./qsubs/q_Jeol_$batch_number.sh
  sed -i s/num_batches/"$num_batches"/g    ./qsubs/q_Jeol_$batch_number.sh
  sed -i s/batch_number/"$batch_number"/g  ./qsubs/q_Jeol_$batch_number.sh
  let "batch_number+=1" 
done