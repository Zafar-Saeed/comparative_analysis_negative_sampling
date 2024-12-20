#!/bin/bash
main_working_directory="/lustrehome/zafarsaeed/source_codes/comparative_analysis_Condor"

experiment_specs=$1
dataset_path=$2
experiment_name=$3
dataset_name=$4

cd $main_working_directory

eval "$(conda shell.bash hook)"
# source /lustrehome/zafarsaeed/.conda/envs/myenv/bin/activate
conda activate myenv
# python3 run_experiments.py experiment_specs ./data/ transe_1 --test_code --test_dataset_name kinships
# python3 run_experiments.py $experiment_specs $dataset_path $experiment_name $dataset_name
# python3 run_experiments.py $experiment_specs $dataset_path $experiment_name --test_code --test_dataset_name $dataset_name
python3 run_experiments.py $experiment_specs $dataset_path $experiment_name $dataset_name