import networkx as nx
import os
import copy
import pickle
import scipy

import numpy as np
import pandas as pd

from tqdm import tqdm

from pathlib import Path


class LoadData():
    '''
    Prepare dictionaries of drugs, indications, proteins, and biological functions
    ID system:
    - indication: UMLS CUID
    - drug: DrugBank ID
    - protein: Entrez ID
    - biological function: GO term
    '''
    def __init__(self):
        abs_path = Path(__file__).parent.absolute()
        self.df_path_dict = {'drug': abs_path / 'data/raw/1_drug_to_protein.tsv',
                             'indication': abs_path / 'data/raw/2_indication_to_protein.tsv',
                             'protein': abs_path / 'data/raw/3_protein_to_protein.tsv',
                             'biological_function': abs_path / 'data/raw/5_biological_function_to_biological_function.tsv'}

    def _constitute_dictionary(self, list_id, list_name, dict_=dict()):
        for i in range(len(list_id)):
            id_ = list_id[i]
            if id_ not in dict_:
                dict_[id_] = list_name[i]
        return dict_
    
    def get_dict(self, type='drug'):
        if type == 'drug' or type == 'indication':
            df = pd.read_csv(self.df_path_dict[type], sep='\t')
            type_name, type_id = df['node_1_name'].tolist(), df['node_1'].tolist()
            type_id2name = self._constitute_dictionary(type_id, type_name, dict_=dict())
            type_name2id = {v : k for k, v in type_id2name.items()}
            return type_id2name, type_name2id
        elif type == 'protein' or type == 'biological_function':
            df = pd.read_csv(self.df_path_dict[type], sep='\t')
            type_name_1, type_id_1 = df['node_1_name'].tolist(), df['node_1'].map(str).tolist()
            type_name_2, type_id_2 = df['node_2_name'].tolist(), df['node_2'].map(str).tolist()
            type_id2name = self._constitute_dictionary(type_id_1, type_name_1, dict_={})
            type_id2name = self._constitute_dictionary(type_id_2, type_name_2, dict_=type_id2name)
            type_name2id = {v : k for k, v in type_id2name.items()}
            return type_id2name, type_name2id
        else:
            print("Wrong type!")
