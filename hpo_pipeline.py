import pykeen
import pandas as pd
from pykeen.hpo import hpo_pipeline
from optuna.samplers import RandomSampler, TPESampler, GridSampler
from pykeen.triples import TriplesFactory
from pykeen.evaluation import RankBasedEvaluator
import os
from typing import Mapping
from data_loader import load_data
from optuna.samplers import GridSampler, RandomSampler


def optimize_hyper_parameters(root_path, dataset_name, corruption_technique, model_name, num_of_negs, num_of_epochs = 100, num_of_tials = 3, overwrite_flag=False):


    train_triples, validation_triples, test_triples = load_data(root_path, dataset_name)

# region HPO GridSampler code from pykeen documentation    
    
    # from pykeen.hpo import hpo_pipeline
    # from optuna.samplers import GridSampler
    # hpo_pipeline_result = hpo_pipeline(
    #     n_trials=30,
    #     sampler=GridSampler,
    #     sampler_kwargs=dict(
    #         search_space={
    #             "model.embedding_dim": [32, 64, 128],
    #             "model.scoring_fct_norm": [1, 2],
    #             "loss.margin": [1.0],
    #             "optimizer.lr": [1.0e-03],
    #             "negative_sampler.num_negs_per_pos": [32],
    #             "training.num_epochs": [100],
    #             "training.batch_size": [128],
    #         },
    #     ),
    #     dataset='Nations',
    #     model='TransE',
    # )
# endregion 

 
    # Run hyperparameter optimization with early stopping and evaluation kwargs
    
    hpo_result = hpo_pipeline(

        #sampler=GridSampler, # sampler for HPO search space other choices = RandomSampler and TPESampler with TPE sampler = TPESampler(prior_weight=1.1)
        # sampler=RandomSampler,
        sampler=TPESampler(),

        n_trials=num_of_tials,  # Number of hyperparameter trials

        training=train_triples,
        testing=test_triples,
        validation=validation_triples,
        
        # Model and defining a search space for embedding dimensions
        # TODO: reading the model name from configuration file and set it here 
        model=model_name,
        model_kwargs_ranges=dict(
            # embedding_dim=dict(type=int, scale='power', base=2, low=32, high=256), # THIS DOES NOT WORK, THROWS AN EXCEPTION, HOWEVER, PyKeen DOCS EXPLAINS SIMILAR SCALING
            embedding_dim=dict(type=int, low=50, high=200, q=50),
        ),
        
        # TODO: Check if I can exclude this margin ranking loss from HPO
        loss='marginranking',
        loss_kwargs_ranges=dict(
            margin=dict(type=float, low=0.5, high=2.0, step=0.5),  # CROSS CHECK IF "step" parameter is correct
        ),
        
        # Training loop to use
        training_loop='sLCWA',  # Choices are 'LCWA', 'sLCWA'
        training_kwargs=dict(num_epochs=num_of_epochs, use_tqdm_batch=False),
        training_kwargs_ranges = dict(
            # batch_size=dict(type=int, low=128, high=1024, scale='power', base=2), # THE POWER SCALE DOES NOT WORK 
            batch_size=dict(type=int, low=200, high=1000, q=200),
        ),
        # Negative sampler to use
        # TODO: read the sampler name and num_negs_per_pos from configuration file
        # TODO: if the sampler is custom, then create an object of Wrapper sampler and set the parameters accordingly, otherwise find a way to initialize the parameters
        negative_sampler=corruption_technique,
        negative_sampler_kwargs=dict(num_negs_per_pos=num_of_negs), #corruption_scheme=('h', 't'), # it is also a defult scheme, this could be removed
            
        # Optimizer and defining a search space for the learning rate in optimizer
        optimizer='adam',  # Choices include 'adam', 'sgd', etc.
        optimizer_kwargs_ranges=dict(
            lr=dict(type=float, low=0.0001, high=0.1, log=True),
        ),
        
        # Regularizer and defining a search space for the L2 regularization weight
        regularizer='lp',  # Choices include 'no', 'lp', 'powerful'
        regularizer_kwargs = dict(p=2),
        regularizer_kwargs_ranges=dict(
            weight=dict(type=float, low=1e-6, high=1e-2, log=True),
        ),
        
        # Evaluator
        evaluator='rankbased',  # Choices include 'rankbased', 'classification'
        evaluator_kwargs=dict(filtered=True),  # Perform filtered evaluation
        evaluation_kwargs=dict(batch_size=1024),
        
        # Early stopper callback
        stopper='early',
        stopper_kwargs=dict(frequency=2, patience=5, relative_delta=0.002),
    )

    # Save the best model and results
    directorty_path = os.path.join(root_path, "hpo_results", dataset_name,corruption_technique)
    os.makedirs(directorty_path,exist_ok=True)

    hpo_directory_path = os.path.join(directorty_path, model_name+"_{}".format(num_of_negs))
    hpo_result.save_to_directory(hpo_directory_path)

    # # Get the best model and hyperparameters
    # THIS hpo_result DOES NOT HAVE MODEL OBJECT, THIS CODE DOES NOT WORK
    # best_model = hpo_result.model

    best_hyperparameters = hpo_result.study.best_params

    # best_trial_params = {}
    # for param_name, param_value in best_hyperparameters.items():
    #     print(f"{param_name}: {param_value}")
    #     best_trial_params[param_name] = param_value
    print(best_hyperparameters)

    import json
    with open(hpo_directory_path+'/best_trial.json', 'w') as json_file:
        json.dump(best_hyperparameters, json_file)

    # Print the best hyperparameters
    print('Best Hyperparameters:', best_hyperparameters)




# root_path = "./data"
# dataset_name = "kinships" 
# corruption_technique = "basic"
# model_name = 'transe'
# num_of_negs = "1"
# num_of_epochs = 10
# num_of_tials = 3
# overwrite_flag = False
# optimize_hyper_parameters(root_path,dataset_name, corruption_technique, model_name, int(num_of_negs), num_of_epochs, num_of_tials, overwrite_flag)

import argparse

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('root_path')
    parser.add_argument('dataset_name')
    parser.add_argument('corruption_technique')
    parser.add_argument('model_name')
    parser.add_argument('num_of_negs')
    parser.add_argument('num_of_epochs')
    parser.add_argument('num_of_tials')
    
    parser.add_argument('--ow', dest='overwrite_flag', action='store_true', help='A boolean flag')
    
    args = parser.parse_args()

    optimize_hyper_parameters(args.root_path, args.dataset_name,args.corruption_technique,args.model_name,int(args.num_of_negs), int(args.num_of_epochs), int(args.num_of_tials))