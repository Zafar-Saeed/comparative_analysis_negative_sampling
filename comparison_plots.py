import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib
import numpy as np

def plot_result_files(root_folder=None):

    root_folder = "./data/compiled_results"
    if root_folder.startswith("."):
        root_folder = os.path.abspath(root_folder)

    matplotlib.use('Agg')
    mrr_results_files = dict()
    print("*** Drawing comparison plots ***")
    for root, dirs, files in os.walk(root_folder):
        dirs[:] = [d for d in dirs if not d.startswith('.')]
            
            # leaf folder
        if not bool(dirs):

            rel_path = os.path.relpath(root, start=root_folder)
            folder_hierarchy = rel_path.split(os.sep) 
            pass
            dataset_name = folder_hierarchy[0]
            #corruption_technique_name = folder_hierarchy[1]
            
            for file_name in files:
                if file_name.endswith("_MRR.csv"):
                    embedding_technique = file_name.split('_')[0]
                    print(folder_hierarchy)
                    draw_plots(embedding_technique, os.path.join(root,file_name))
    
    print("*** Completed ***")
               
def draw_plots(embedding_technique, file_path):
   
    # Read the CSV file
    df = pd.read_csv(file_path)

    # Extract the corruption technique column
    # techniques = df['corruption']
    column_name = df.columns[0]
    techniques = df[column_name]
    # Drop the corruption technique column from the dataframe for plotting
    df = df.drop(columns=[column_name])

    # Convert the remaining header columns to integers
    df.columns = [int(col) for col in df.columns]

    # Plot the line chart
    plt.figure(figsize=(10, 6))
    for index, row in df.iterrows():
        plt.plot(df.columns, row, marker='o', label=techniques[index])

    plt.xscale('log')
    plt.xlabel('No. of Negatives')
    plt.ylabel('Mean Reciprocal Rank')
    plt.title(embedding_technique)
    plt.yticks(np.arange(0, 1.1, 0.2))
    plt.ylim(0, 1)
    plt.xlim(1, 100)
    plt.legend(frameon=False)
    plt.grid(True, color='#f7f7f7')

    # Get the absolute path of the file
    abs_file_path = os.path.abspath(file_path)
    # Get the directory containing the file
    file_directory = os.path.dirname(abs_file_path)
    # Get the parent directory of the directory containing the file
    parent_directory = os.path.dirname(file_directory)

    # charts_path = os.path.join(parent_directory,file_directory+"_Plots")
    os.makedirs(file_directory,exist_ok=True)
    plt.savefig(os.path.join(file_directory, embedding_technique+"_MRR.pdf"))
    #plt.show()
    plt.close()

plot_result_files("./data/compiled_results")
# import argparse

# if __name__=='__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('data_path')
    
#     args = parser.parse_args()
#     # args.data_path = "./data/compiled_results"
#     manage_result_files(args.data_path)
