#!/bin/bash
cd /home/a.norouzikandelati/Projects/Double_Crop_Mapping/

for block_count in 20
do 
  batch_number=1
  while [ $batch_number -le 20 ]
  do
    cp template.sh                           ./qsubs/q_$batch_number.sh
    sed -i s/block_count/"$block_count"/g    ./qsubs/q_$batch_number.sh
    sed -i s/batch_number/"$batch_number"/g  ./qsubs/q_$batch_number.sh
  done
done  