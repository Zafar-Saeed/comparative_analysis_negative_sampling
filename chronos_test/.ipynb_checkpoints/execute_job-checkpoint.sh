#!/bin/bash

pwd
cd ..
eval "$(conda shell.bash hook)"
# source /lustrehome/zafarsaeed/.conda/envs/myenv/bin/activate
conda activate myenv
python3 run_experiments.py experiment_specs ./data/ transe_1 --test_code --test_dataset_name kinships
