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
missing_numbers=(527 528 552 560 561 568 593 594 605 606 609 677 687 690 700 701 711 764 770 775 776 783 784 791 793 834 864 908 934 976)

# Submit jobs for missing batch numbers
for batch_number in "${missing_numbers[@]}"
do
  sbatch ./q_Jeol_$batch_number.sh
done
