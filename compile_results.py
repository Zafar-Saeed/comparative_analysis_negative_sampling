
import os
import re
import json
import csv
from util import write_csv


def init_defult_result_dict(type_header: str, corruption_technique: str):
    return {
        type_header: corruption_technique,
        '1': 0,
        '2': 0,
        '5': 0,
        '10': 0,
        '20': 0,
        '50': 0,
        '100': 0,
    }

def manage_result_files(root_folder=None):
    
    #temporary, later remove this when merged into comparative_analysis project
    
    # if root_folder.startswith("."):
    #     root_folder = os.path.abspath(root_folder)

    #root_folder = "/Users/zafarsaeed/Uniba Italy/Research/Source Code/code/comparative_analysis/data/Results"

    # header = ['corruption', '1', '2', '5', '10', '50', '100']
    embedding_model = set()
    num_corruption_sample = set()
    model_files = dict()
    dataset_name = ""
    corruption_technique_name = ""
    arranged_results_files = dict()

    for root, dirs, files in os.walk(root_folder):

        # Skip hidden directories
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        
        # leaf folder
        if not bool(dirs):

            rel_path = os.path.relpath(root, start=root_folder)
            folder_hierarchy = rel_path.split(os.sep) 

            dataset_name = folder_hierarchy[0]
            corruption_technique_name = folder_hierarchy[1]
            
            # for file_name in files:
            #     file_name_parts = file_name.split('_')
            #     embedding_model.add(file_name_parts[0]) # model name
            #     if ".model" not in file_name_parts[1]:
            #         num_corruption_sample.add(file_name_parts[1]) # number of corruption samples
            for file_name in (f for f in files if "tucker" not in f and f.endswith(".json")): # removing tucker explicity, because for some techniques tucker results are availabel, later we decided to remove tucker reason being non-scalable
                file_name_parts = file_name.split('_')
                embedding_model.add(file_name_parts[0])
                num_corruption_sample.add(file_name_parts[1])

            for model_name in embedding_model:
                model_result_files = [os.path.join(root,file) for file in files if file.startswith(model_name) and file.endswith("metrics_hit_mrr.json")] # get all the result files of each model
                model_losses_files = [os.path.join(root,file) for file in files if file.startswith(model_name) and file.endswith("losses.json")] # get all the losses files of each model

                # mode_file_list = set(model_files.get(model, set()))
                if model_name not in model_files:
                    model_files[model_name] = {
                        corruption_technique_name: model_result_files+model_losses_files
                    }
                else:
                    corruption_techniques = model_files[model_name]
                    corruption_techniques[corruption_technique_name] = model_result_files+model_losses_files
            if dataset_name not in arranged_results_files:
                arranged_results_files[dataset_name] = model_files
            else:
                dataset_model_files = arranged_results_files[dataset_name]
                if set(dataset_model_files.keys()) == set(model_files.keys()):
                    for key in dataset_model_files:
                        corruption_technique_files = model_files[key]
                        first_key = next(iter(corruption_technique_files))
                        dataset_model_files[key][first_key] = corruption_technique_files[first_key]
                else:
                    raise Exception("Models are not same in dataset_model_files and mode_files...")
            model_files = dict()
    return arranged_results_files

def compile_results(root_folder: str = None):
    result_files = manage_result_files(root_folder)

    data_mmr = []
    data_hit1 = []
    data_hit3 = []
    data_hit5 = []
    data_hit10 = []
    data_exe_time = []
    
    #temporary, later remove this when merged into comparative_analysis project
    # root_folder = "/Users/zafarsaeed/Uniba Italy/Research/Source Code/code/comparative_analysis/data/Results"

    results_path = os.path.join(os.path.dirname(root_folder),"compiled_results")

    for dataset in result_files:
        print("**** Compiling embedding model based results for dataset: {} ****".format(dataset))
        dataset_results_files = result_files[dataset]
        for model in dataset_results_files:
            model_results_files = dataset_results_files[model]
            for corruption_technique in model_results_files:
                file_list = model_results_files[corruption_technique]

                metrics_files = [file for file in file_list if file.endswith("metrics_hit_mrr.json")]
                losses_files = [file for file in file_list if file.endswith("losses.json")]

                results_mmr = init_defult_result_dict("negative_sampling", corruption_technique)
                results_hit1 = init_defult_result_dict("negative_sampling",corruption_technique)
                results_hit3 = init_defult_result_dict("negative_sampling",corruption_technique)
                results_hit5 = init_defult_result_dict("negative_sampling",corruption_technique)
                results_hit10 = init_defult_result_dict("negative_sampling",corruption_technique)
                results_exe_time = init_defult_result_dict("negative_sampling",corruption_technique)
                
                for file in metrics_files:
                    result_file = json.load(open(file))

                    start_marker = model+"_"
                    end_marker = "_metrics_hit_mrr.json"
                    
                    start_index = file.find(start_marker)
                    end_index = file.find(end_marker, start_index + len(start_marker))
                    num_of_negs = file[start_index + len(start_marker):end_index].strip()
                    results_mmr[num_of_negs] = result_file['MRR']
                    results_hit1[num_of_negs] = result_file['hit_at_1']
                    results_hit3[num_of_negs] = result_file['hit_at_3']
                    results_hit5[num_of_negs] = result_file['hit_at_5']
                    results_hit10[num_of_negs] = result_file['hit_at_10']

                    # need to fix the execute time in main project, it has to be within a certain format
                    results_exe_time[num_of_negs] = result_file['training_time']

                data_mmr.append(results_mmr)
                data_hit1.append(results_hit1)
                data_hit3.append(results_hit3)
                data_hit5.append(results_hit5)
                data_hit10.append(results_hit10)
                data_exe_time.append(results_exe_time)

                    # add code to generate another file that compres the average exection time for all the corruption techniques
            data_mmr.sort(key=lambda x: x['negative_sampling'])
            data_hit1.sort(key=lambda x: x['negative_sampling'])
            data_hit3.sort(key=lambda x: x['negative_sampling'])
            data_hit5.sort(key=lambda x: x['negative_sampling'])
            data_hit10.sort(key=lambda x: x['negative_sampling'])
            data_exe_time.sort(key=lambda x: x['negative_sampling'])

            write_csv(os.path.join(results_path,dataset, "embedding_model", model),model+"_MRR.csv",data_mmr, "negative_sampling")
            write_csv(os.path.join(results_path,dataset, "embedding_model", model),model+"_hit1.csv",data_hit1, "negative_sampling")
            write_csv(os.path.join(results_path,dataset, "embedding_model", model),model+"_hit3.csv",data_hit3, "negative_sampling")
            write_csv(os.path.join(results_path,dataset, "embedding_model", model),model+"_hit5.csv",data_hit5, "negative_sampling")
            write_csv(os.path.join(results_path,dataset, "embedding_model", model),model+"_hit10.csv",data_hit10, "negative_sampling")
            write_csv(os.path.join(results_path,dataset, "embedding_model", model),model+"_ExeTime.csv",data_exe_time, "negative_sampling")
            data_mmr = []
            data_hit1 = []
            data_hit3 = []
            data_hit5 = []
            data_hit10 = []
            data_exe_time = []

    ns_results_orientation(root_folder)
    return True

def ns_results_orientation(root_folder=None):
 
    embedding_model = set()
    num_corruption_sample = set()
    ns_files = dict()
    dataset_name = ""
    corruption_technique_name = ""
    arranged_results_files = dict()

    data_mmr = []
    data_hit1 = []
    data_hit3 = []
    data_hit5 = []
    data_hit10 = []
    data_exe_time = []
    
    results_path = os.path.join(os.path.dirname(root_folder),"compiled_results")

    for root, dirs, files in os.walk(root_folder):

        # Skip hidden directories
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        
        # leaf folder
        if not bool(dirs):

            rel_path = os.path.relpath(root, start=root_folder)
            folder_hierarchy = rel_path.split(os.sep) 

            dataset_name = folder_hierarchy[0]
            corruption_technique_name = folder_hierarchy[1]
            
            for file_name in (f for f in files if "tucker" not in f and f.endswith(".json")): # removing tucker explicity, because for some techniques tucker results are availabel, later we decided to remove tucker reason being non-scalable
                file_name_parts = file_name.split('_')
                embedding_model.add(file_name_parts[0])
                num_corruption_sample.add(file_name_parts[1])

            for model_name in embedding_model:
                model_result_files = [os.path.join(root,file) for file in files if file.startswith(model_name) and file.endswith("metrics_hit_mrr.json")] # get all the result files of each model
                model_losses_files = [os.path.join(root,file) for file in files if file.startswith(model_name) and file.endswith("losses.json")] # get all the losses files of each model

                results_mmr = init_defult_result_dict("embedding_model", model_name)
                results_hit1 = init_defult_result_dict("embedding_model", model_name)
                results_hit3 = init_defult_result_dict("embedding_model", model_name)
                results_hit5 = init_defult_result_dict("embedding_model", model_name)
                results_hit10 = init_defult_result_dict("embedding_model", model_name)
                results_exe_time = init_defult_result_dict("embedding_model", model_name)
                
                for file in model_result_files:
                    result_file = json.load(open(file))

                    start_marker = model_name+"_"
                    end_marker = "_metrics_hit_mrr.json"
                    
                    start_index = file.find(start_marker)
                    end_index = file.find(end_marker, start_index + len(start_marker))
                    num_of_negs = file[start_index + len(start_marker):end_index].strip()
                    results_mmr[num_of_negs] = result_file['MRR']
                    results_hit1[num_of_negs] = result_file['hit_at_1']
                    results_hit3[num_of_negs] = result_file['hit_at_3']
                    results_hit5[num_of_negs] = result_file['hit_at_5']
                    results_hit10[num_of_negs] = result_file['hit_at_10']

                    # need to fix the execute time in main project, it has to be within a certain format
                    if  not isinstance(result_file['training_time'], float) and  ":" in result_file['training_time']:
                        training_time_parts = result_file['training_time'].split(':')
                        total_training_seconds = int(float(training_time_parts[0]))*3600 + int(float(training_time_parts[1]))*60 + int(float(training_time_parts[2]))
                    results_exe_time[num_of_negs] = total_training_seconds

                data_mmr.append(results_mmr)
                data_hit1.append(results_hit1)
                data_hit3.append(results_hit3)
                data_hit5.append(results_hit5)
                data_hit10.append(results_hit10)
                data_exe_time.append(results_exe_time)
            
            data_mmr.sort(key=lambda x: x['embedding_model'])
            data_hit1.sort(key=lambda x: x['embedding_model'])
            data_hit3.sort(key=lambda x: x['embedding_model'])
            data_hit5.sort(key=lambda x: x['embedding_model'])
            data_hit10.sort(key=lambda x: x['embedding_model'])
            data_exe_time.sort(key=lambda x: x['embedding_model'])

            write_csv(os.path.join(results_path,dataset_name, "negative_sampling", corruption_technique_name),corruption_technique_name+"_MRR.csv",data_mmr, "embedding_model")
            write_csv(os.path.join(results_path,dataset_name, "negative_sampling", corruption_technique_name),corruption_technique_name+"_hit1.csv",data_hit1, "embedding_model")
            write_csv(os.path.join(results_path,dataset_name, "negative_sampling", corruption_technique_name),corruption_technique_name+"_hit3.csv",data_hit3, "embedding_model")
            write_csv(os.path.join(results_path,dataset_name, "negative_sampling", corruption_technique_name),corruption_technique_name+"_hit5.csv",data_hit5, "embedding_model")
            write_csv(os.path.join(results_path,dataset_name, "negative_sampling", corruption_technique_name),corruption_technique_name+"_hit10.csv",data_hit10, "embedding_model")
            write_csv(os.path.join(results_path,dataset_name, "negative_sampling", corruption_technique_name),corruption_technique_name+"_ExeTime.csv",data_exe_time, "embedding_model")
            data_mmr = []
            data_hit1 = []
            data_hit3 = []
            data_hit5 = []
            data_hit10 = []
            data_exe_time = []

    return True

# compile_results("./data/Results")
import argparse

if __name__=='__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('data_path')
    
    # args = parser.parse_args()

    # compile_results(args.data_path)
    compile_results("./data/results")
