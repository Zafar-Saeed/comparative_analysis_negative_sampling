import os
import pickle as pickle
import constants
import json
from pykeen.datasets import get_dataset
import re
import sys
import csv
# class Index(object):
#     def __init__(self):
#         self.ent_index = dict()
#         self.rel_index = dict()

#     def rel_to_ind(self, rel):
#         if rel not in self.rel_index:
#             self.rel_index[rel] = len(self.rel_index.keys())
#         return self.rel_index[rel]

#     def ent_to_ind(self, ent):
#         if ent not in self.ent_index:
#             self.ent_index[ent] = len(self.ent_index.keys())
#         return self.ent_index[ent]

#     def load_index(self,dir_name) -> bool:
#         if os.path.exists(os.path.join(dir_name,constants.entity_ind_file)) and os.path.exists(os.path.join(dir_name,constants.rel_ind_file)):
#             self.ent_index = pickle.load(open(os.path.join(dir_name,constants.entity_ind_file),'rb'))
#             self.rel_index = pickle.load(open(os.path.join(dir_name, constants.rel_ind_file),'rb'))
#             return True
#         return False
    
#     def save_index(self,dir_name):
#         pickle.dump(self.ent_index,open(os.path.join(dir_name,constants.entity_ind),'wb'))
#         pickle.dump(self.rel_index,open(os.path.join(dir_name, constants.rel_ind), 'wb'))

#     def ent_vocab_size(self):
#         return len(self.ent_index)

#     def rel_vocab_size(self):
#         return len(self.rel_index)
    
#     def create_negative_sample_index(self, dir_name: str):   
#         #directory = './your_folder/'
#         # txt_files = []
#         # for file in os.listdir(dir_name):
#         #     if file.endswith('.txt'):
#         #         txt_files.append(file)
#         all_files = os.listdir(dir_name)
#         txt_files = [file for file in all_files if file.endswith(".txt")]
#         for file in txt_files:
#             with open(os.path.join(dir_name,file), 'r', encoding='UTF-8') as sample_file:
#                 while line := sample_file.readline():
#                     triple = line.split(" ")

def merge_negative_sample_files(directory_path: str, file_prefix: str):
    
    try:
        pkl_files = []
        # Filter the list to include only .pkl files
        pattern = re.compile(f'{file_prefix}\\d+\\.pkl')
        for filename in os.listdir(directory_path):
            if pattern.match(filename):
                pkl_files.append(filename)

        pkl_files = sorted(pkl_files)
        
        full_ns_dictionary = pickle.load(open(os.path.join(directory_path,pkl_files[0]), 'rb'))
        total_files = len(pkl_files)
        file_no = 1    
        for file in pkl_files[1:]:
            file_no+=1
            ns_dictionary = pickle.load(open(os.path.join(directory_path,file), 'rb'))
            sys.stdout.write("Merging Negative Sample: {}, ... {}% COMPLETED\r".format(file, (int(file_no/total_files*100))))
            full_ns_dictionary = merge_dicts_with_union(full_ns_dictionary,ns_dictionary)
            
        
        return full_ns_dictionary
    
    except FileNotFoundError:
        print(f"Error: The directory '{directory_path}' does not exist.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    return []
   
def merge_with_same_keys(sample_1,sample_2, all_keys):
    
    for key in all_keys:
        # Get values from both dictionaries, default to empty set if key is not present
        ns_samples_1 = sample_1[key]
        ns_samples_2 = sample_2[key]
        
        if isinstance(ns_samples_1, set) and isinstance(ns_samples_2, set):
            ns_samples_1 = set(ns_samples_1)
            ns_samples_2 = set(ns_samples_2)
        else:
            raise Exception("negative samples type is NOT a set()") 
        
        # Perform union of values
        sample_1[key] = ns_samples_1.union(ns_samples_2)

    return sample_1

def merge_with_different_keys(sample_1: dict,sample_2: dict, all_keys):
    
    for key in all_keys:
        # Get values from both dictionaries, default to empty set if key is not present
        ns_samples_1 = sample_1.get(key, set)
        ns_samples_2 = sample_2.get(key, set)
        
        if isinstance(ns_samples_1, set) and isinstance(ns_samples_2, set):
            # ns_samples_1 = set(ns_samples_1)
            # ns_samples_2 = set(ns_samples_2)
            sample_1[key] = ns_samples_1.union(ns_samples_2)
        else:
            raise Exception("negative samples type is NOT a set()") 

        # sample_1[key] = ns_samples_1.union(ns_samples_2)

    return ns_samples_1

def merge_dicts_with_union(sample_1: dict, sample_2: dict):
    
    first_dict_keys = sample_1.keys()
    second_dict_keys = sample_2.keys()
    if first_dict_keys == second_dict_keys:
        return merge_with_same_keys(sample_1,sample_2, first_dict_keys)
    else:
        print("calling: merge_with_different_keys. Keys are not same in sample files")
        return merge_with_different_keys(sample_1,sample_2, set(first_dict_keys.keys()).union(set(second_dict_keys.keys())))
    
def dump_to_pickle_file(data, directory: str, file_name: str):
    print("**** Dumping data to a pickle file using the highest protocol available ****")
    if not os.path.exists(directory):
        print("\t{} does not exist, creating new directory.\n".format(directory))
        os.makedirs(directory)

    if os.path.exists(os.path.join(directory,file_name)):
        print("\tPickel file already exist on path. {0} Overwriting..".format(file_name))
    
    with open(os.path.join(directory,file_name), 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

def dump_json(data, directory: str, file_name: str):
    print("**** Dumping Json file: {} ****".format(file_name))
    if not os.path.exists(directory):
        print("\t{} does not exist, creating new directory.\n".format(directory))
        os.makedirs(directory)
    
    if os.path.exists(os.path.join(directory,file_name)):
        print("\tJson file already exist on path. Now overwriting {0} ...".format(file_name))
    
    json.dump(
        data,
        open(
            os.path.join(
                directory,file_name
            ),
            'w'
        ),
        sort_keys=True,
        separators=(',\n', ': ')
    )

def time_difference(start_time, end_time):
    time_difference = end_time - start_time

    # days = time_difference // (24 * 3600)
    # time_difference %= (24 * 3600)
    hours = time_difference // 3600
    time_difference %= 3600
    minutes = time_difference // 60
    seconds = time_difference % 60

    return hours,minutes,seconds

def time_difference_in_seconds(start_time, end_time):
    time_difference = end_time - start_time

    return time_difference

def write_triples_with_labels_to_file(triples, entity_label_map, relation_label_map, file_path):
    with open(file_path, 'w') as f:
        for head_id, relation_id, tail_id in triples:
            head_label = entity_label_map[head_id]
            relation_label = relation_label_map[relation_id]
            tail_label = entity_label_map[tail_id]
            f.write(f"{head_label}\t{relation_label}\t{tail_label}\n")

def write_pykeen_dataset(data_path: str, dataset_name: str):
    # Datasets offered by Pykeen:
    # Valid choices are: ['aristov4', 'biokg', 'ckg', 'cn3l', 'codexlarge', 'codexmedium', 'codexsmall', 'conceptnet', 
    #                     'countries', 'cskg', 'db100k', 'dbpedia50', 'drkg', 'fb15k', 'fb15k237', 'globi', 'hetionet', 
    #                     'kinships', 'nations', 'nationsliteral', 'ogbbiokg', 'ogbwikikg2', 'openbiolink', 'openbiolinklq', 
    #                     'openea', 'pharmebinet', 'pharmkg', 'pharmkg8k', 'primekg', 'umls', 'wd50kt', 'wikidata5m', 'wk3l120k', 
    #                     'wk3l15k', 'wn18', 'wn18rr', 'yago310']"
    train_data_file = os.path.join(data_path, dataset_name, "train")
    valid_data_file = os.path.join(data_path, dataset_name, "dev")
    test_data_file = os.path.join(data_path, dataset_name, "test")

    if not (os.path.exists(train_data_file) and os.path.exists(valid_data_file) and os.path.exists(test_data_file)):
        dataset = get_dataset(dataset=dataset_name) # this might throw either an exception of return empty dataset because  of mismatching name OR Pykeen do not support dataset with provided nam  

        training_triples = dataset.training.mapped_triples.tolist()
        validation_triples = dataset.validation.mapped_triples.tolist()
        testing_triples = dataset.testing.mapped_triples.tolist()

        # Create mappings from IDs to labels
        entity_label_map = {v: k for k, v in dataset.entity_to_id.items()}
        relation_label_map = {v: k for k, v in dataset.relation_to_id.items()}

        os.makedirs(os.path.join(data_path, dataset_name),exist_ok=True)
        
        write_triples_with_labels_to_file(training_triples, entity_label_map, relation_label_map, train_data_file)
        write_triples_with_labels_to_file(validation_triples, entity_label_map, relation_label_map, valid_data_file)
        write_triples_with_labels_to_file(testing_triples, entity_label_map, relation_label_map, test_data_file)

def load_model(model_name, config):
    return model_name

def load_config_files(directory_path: str, extension: str):
    
    try:
        if os.path.exists(directory_path):

            config_files = []
            # Filter the list of files include only *.json
            pattern = re.compile(r'.*\.{}$'.format(extension), re.IGNORECASE)
            for filename in os.listdir(directory_path):
                if pattern.match(filename):
                    config_files.append(filename)

            config_files = sorted(config_files)
            
            config = json.load(open(os.path.join(directory_path,config_files[0])))    
            all_config_files = dict()

            all_config_files[config_files[0]] = config

            total_files = len(config_files)
            print("\tConfiguring {} number of experiments...".format(total_files))   
            for file in config_files[1:]:
                print("\t\tLoading configuration files: {}".format(file))
                all_config_files[file] = json.load(open(os.path.join(directory_path,file)))
               
            return all_config_files
        else:
            return None
    
    except FileNotFoundError:
        print(f"Error: The directory '{directory_path}' does not exist.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    return None

def write_csv(directory_path: str, file_name: str, data: list, type_header: str):
    # create directory if it does not exist
    os.makedirs(directory_path,exist_ok=True)
    print("\twriting results in csv file: {}".format(file_name))

    if os.path.isfile(os.path.join(directory_path,file_name)):
        print("\tFile {} already exists.".format(file_name))
    else:
        with open(os.path.join(directory_path,file_name), 'w', newline='') as csvfile:
            fieldnames = [type_header, '1', '2', '5', '10', '20', '50', '100']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)


def write_csv_ns_count(data: dict, directory_path: str, file_name: str, is_target):

    corrupt_side = "targets" if is_target else "sources"
    
    f_name = "{}_{}.csv".format(file_name, corrupt_side)
    print("\twriting negative {} count for each entity. File name: {}".format(corrupt_side,f_name))

    # sorted_data = {k: data[corrupt_side][k] for k in sorted(data[corrupt_side], key=data[corrupt_side].get)}

    # sorted_data = dict(sorted(data.items(), key=lambda item: item[1]))

    with open(os.path.join(directory_path,f_name), mode='w', newline='') as file:
        writer = csv.writer(file, delimiter=',')    
        writer.writerow(['Entity', 'Max-Neg.'])  # Write header
        # writer.writerows(sorted_data.items())  # Write key-value pairs
        processed_data = dict()
        
        for key in data[corrupt_side].keys():
            processed_data[key] = str(len(data[corrupt_side][key]))
        processed_data = dict(sorted(processed_data.items(), key=lambda item: item[1]))
        value_list = list(processed_data.values())
        sample_count = list()
        for ns in range(0,101):
            sample_count.append(value_list.count(str(ns)))
        
        auxilary_samples_required = dict()
        for i in range(1,len(sample_count)+1):
            seq = 1
            counting = 0
            for num in range(i-1,-1,-1):
                counting += sample_count[num]*(seq)
                seq += 1

            auxilary_samples_required[i] = counting

        for key in data[corrupt_side].keys():
            # writer.writerows(data[corrupt_side].items())  # Write key-value pairs
            writer.writerow([str(key), str(len(data[corrupt_side][key]))])  # Write key-value pairs
        
        writer.writerow(["No. of Neg.","Auxilary"])

        for key in auxilary_samples_required.keys():
            # writer.writerows(data[corrupt_side].items())  # Write key-value pairs
            writer.writerow([str(key), str(auxilary_samples_required[key])])  # Write key-value pairs