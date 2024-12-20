from typing import Mapping
import os
from shared_index import SharedIndex
from pykeen.triples import TriplesFactory
import torch


def get_files(directory_path, post_fix):
    files = os.listdir(directory_path)
    model_files = [file for file in files if file.endswith(post_fix)]
    return model_files

def load_data(root_folder: str, dataset_name: str = "kinships", is_dev_mode: bool = False):
    # dataset_name = "kinships" # testing the code
    train_file = os.path.join(root_folder,dataset_name, "train")
    valid_file = os.path.join(root_folder,dataset_name, "dev")
    test_file = os.path.join(root_folder,dataset_name, "test")

    # shared_index = Index()
    # shared_index.load_index(os.path.join(root_folder,dataset_name))
    if os.path.exists(os.path.join(root_folder,dataset_name)):

        shared_index = SharedIndex.get_instance()
        shared_index.index.load_index(os.path.join(root_folder,dataset_name))

        EntityMapping = Mapping[str, int]
        RelationMapping = Mapping[str, int]

        # entity_mapping: EntityMapping = shared_index.ent_index
        # relation_mapping: RelationMapping = shared_index.rel_index
        entity_mapping: EntityMapping = shared_index.index.ent_index
        relation_mapping: RelationMapping = shared_index.index.rel_index

        # Read the triples from the text files
        print("Loading training tripples..")
        train_triples = TriplesFactory.from_path(
            path=train_file, 
            entity_to_id=entity_mapping, 
            relation_to_id=relation_mapping)

        print("Loading validation tripples..")
        validation_triples = TriplesFactory.from_path(
            path=valid_file, 
            entity_to_id=entity_mapping, 
            relation_to_id=relation_mapping)

        print("Loading test tripples..")
        test_triples = TriplesFactory.from_path(
            path=test_file, 
            entity_to_id=entity_mapping, 
            relation_to_id=relation_mapping)

        #MERGING TWO TRIPPLES FACTORIES
        train_triples_array = train_triples.mapped_triples

        if is_dev_mode:
            test_triples_array = validation_triples.mapped_triples
        else:
            test_triples_array = test_triples.mapped_triples

        # Concatenate the two sets of triples
        merged_triples = torch.cat([train_triples_array, test_triples_array], dim=0)

        # Create a new TriplesFactory from the merged triples
        neg_sampler_triples = TriplesFactory(
            mapped_triples=merged_triples,
            entity_to_id=entity_mapping,  # Entity mapping from the training triples
            relation_to_id=relation_mapping  # Relation mapping from the training triples
        )

        return train_triples, validation_triples, test_triples, neg_sampler_triples
    else:
        raise Exception("Path does not exist: {}".format(os.path.join(root_folder,dataset_name)))