# #!/bin/bash
# cd /home/a.norouzikandelati/Projects/Tillage_mapping/codes/qsubs

# # Run jobs from batch_number to ..
# batch_number=901
# while [ $batch_number -le 1000 ]
# do
#   sbatch ./q_Jeol_$batch_number.sh
#   let "batch_number+=1"
# done


# Run missing batch numbers
#!/bin/bash
cd /home/a.norouzikandelati/Projects/Tillage_mapping/codes/qsubs

# Define missing batch numbers (your list)
missing_numbers=(1000)

# Submit jobs for missing batch numbers
for batch_number in "${missing_numbers[@]}"
do
  sbatch ./q_Jeol_$batch_number.sh
done
