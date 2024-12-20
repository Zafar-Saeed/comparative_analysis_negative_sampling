#!/bin/bash

main_working_directory="/lustrehome/zafarsaeed/source_codes/comparative_analysis_Condor"
exp_config_folder="experiment_specs"
dataset_folder="data"
dataset_name="FB15K"
# dataset_name="kinships"
exp_config_directory_path="${main_working_directory}/${dataset_folder}/${exp_config_folder}"
condor_directory_path="./${dataset_name}"

condor_logs_path="${condor_directory_path}/logs"
condor_config_path="${condor_directory_path}/job_configs"

# Check if the directory does not exist, then create it
if [ ! -d "$condor_directory_path" ]; then
    mkdir -p $condor_logs_path
    mkdir -p $condor_config_path  
    echo "Directories created for log and job files: $condor_directory_path"
else
    # cleaning everything inside directory_path and re-creating fresh folders
    rm -rf "$condor_directory_path"/*
    mkdir -p $condor_logs_path
    mkdir -p $condor_config_path  
    echo "Cleaned previsous log and job files at: $condor_directory_path"
fi


for experiment_config_file in "${exp_config_directory_path}"/*.json;
do
    # Strip the .json extension and print the file name
    experiment_name="$(basename $experiment_config_file)"
    experiment_name="${experiment_name%.json}"
    
    echo "Training model name: $experiment_name"
    if [ "$experiment_name" == "rescal_100" ] || [ "$experiment_name" == "rescal_50" ] || [ "$experiment_name" == "rescal_20" ]; then
        required_memory="32 GB"
    else
        required_memory="8 GB"
    fi

    content="universe = vanilla
    executable = execute_job.sh
    arguments = "$exp_config_folder" "./${dataset_folder}" "$experiment_name" "$dataset_name"
    output = ${condor_logs_path}/job_${experiment_name}.out
    error = ${condor_logs_path}/job_${experiment_name}.error
    log = ${condor_logs_path}/job_${experiment_name}.log
    request_cpus = 1
    request_gpus = 1
    request_memory = ${required_memory}
    rank = Memory
    queue"

    echo "$content" > "${condor_config_path}/${experiment_name}"

    # echo "condor_submit "${condor_config_path}/${experiment_name}" -name ettore"
    condor_submit "${condor_config_path}/${experiment_name}" -name ettore
    
done