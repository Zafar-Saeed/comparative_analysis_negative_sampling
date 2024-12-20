import numpy as np
import json
import constants
import os
import datetime as dt
import util
def main():
    data = 'FB15K'
    
    #every time a new folder will be created
    base = "./data/experiment_configs_{}_{}".format(data,dt.datetime.now().strftime("%Y-%m-%d"))

    # Later, I need to change the model configuration with hyper parameter optimization
    default_regularizer_value = 0.00001 
   
# region Regularizer l2 values for Corrupt, Relational, Typed, Adverserial, NN
    # region WordNet dataset
    models = {
        'transe':0.0001863777691779108,
        'transh': default_regularizer_value,
        'transr': default_regularizer_value,
        'transd': default_regularizer_value,
        'distmult':3.120071843121878e-06,
        'complex': 2.8198448631731174e-05,
        'simple': 2.8198448631731174e-05,
        'rotate': default_regularizer_value,
        'rescal': 7.484410236920948e-05,
        'tucker': default_regularizer_value,
        'quate': default_regularizer_value,
        'hole': default_regularizer_value,
        'boxe': default_regularizer_value,
        }
# endregion
   
   # region FB15K dataset
    # models = {
    #     # 'transe':0.00024036,
    #     'transh': default_regularizer_value,
    #     'transr': default_regularizer_value,
    #     'transd': default_regularizer_value,
    #     # 'distmult': 4.93E-06,
    #     # 'complex': 1.31E-06,
    #     'simple': default_regularizer_value,
    #     'rotate': default_regularizer_value,
    #     # 'rescal': 0.0002084,
    #     'tucker': default_regularizer_value,
    #     'quate': default_regularizer_value,
    #     'hole': default_regularizer_value,
    #     'boxe': default_regularizer_value,
    #     }
    # endregion
# endregion 

    # at this moment typed information is available for Freebase dataset only
    # later, datasets with schema level information could also be included
    # also, I need to add more techniques in the following array to generates configs
    if data.lower() == 'fb15k' or data.lower() == 'fb15k237': 
        samplers = {"random","corrupt","relational","typed", "nn","adversarial"}
    else:
        samplers = {"random","corrupt","relational", "nn","adversarial"}

    #models = {'complex'}
    l2 = 1.3074905074564395e-06# from hyper-param tuning
    for model,l2 in models.items():
        for sampler in samplers:
            num_negs(model,data,base,l2,sampler)
        #tune_l2(model,data,base)

def num_negs(model,data,base,l2,sampler):
    os.makedirs(base + "/experiment_specs/{}".format(sampler),exist_ok=True)

    # if not os.path.exists(base + "{}/experiment_specs/{}".format(data,sampler)):
    #     os.mkdir(base + "{}/experiment_specs/{}".format(data,sampler))
    path = base + "/experiment_specs/{}/".format(sampler)
    exp_name = "{}".format(model) + "{}.json"
    config = create_config(model,sampler,l2)
    negs = [1,2,5,10,20,50,100]
    for n in negs:
        
        config['num_negs'] = n

        # for WN18 dataset according to the paper
        if data == 'WN18':
            if n < 10:
                config['lr'] = 0.005
            else:
                config['lr'] = 0.01
        elif data == 'FB15K':
            config['lr'] = 0.001
            
        # dump_json(data, directory: str, file_name: str):
        util.dump_json(config, path, exp_name.format("_" + str(n)))
        # json.dump(config, open(path + exp_name.format("_" + str(n)), 'w'),
        #           sort_keys=True, separators=(',\n', ':'))

def tune_l2(model,data,base):
    path = base+"{}/experiment_specs/".format(data)
    exp_name = "{}".format(model) + "{}.json"
    config = create_config(model,'random',0.0)
    l2 = np.sort(np.random.uniform(3.5,6,size=4)) 

    for count,e in enumerate(l2):
            config['l2'] = np.power(10,-e)
            json.dump(config,open(path+exp_name.format("_"+str(count+1)),'w'),
                      sort_keys=True,separators=(',\n', ':'))

def create_config(model_name,neg_sampler,l2):
    config = dict()
    config['model'] = model_name

# region learning rate lr values for Corrupt, Relational, Typed, Adverserial, NN
    # # region FB15K
    # # config['lr'] = 0.001
    # # endregion

    # # region WN18
    # config['lr'] = 0.01
    # # endregion

# endregion
    
    config['l2'] = l2
    config['batch_size'] = constants.batch_size
    config['neg_sampler'] = neg_sampler
    config['num_negs'] = 10
    config['num_epochs']= 100
    config['is_dev'] = False
    config['ent_dim'] = 100
    return config

if __name__=='__main__':
    main()