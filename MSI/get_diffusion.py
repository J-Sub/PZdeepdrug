import networkx as nx
import os
import copy
import pickle
import scipy

import numpy as np
import pandas as pd

from tqdm import tqdm

from typing import List, Tuple

from pathlib import Path

from load_data import LoadData

class Diffusion():
    def __init__(self):
        abs_raw_path = Path(__file__).parent.absolute() / 'data/raw/'
        diff_path = abs_raw_path / '10_top_msi'
        self.path = diff_path
        self.filenames = os.listdir(diff_path)
        self.dataloader = LoadData()
        
        self.idx2node = self.dataloader.load_idx2node()
        self.drug_id2name = self.dataloader.get_dict(type='drug')[0]
        self.indication_id2name = self.dataloader.get_dict(type='indication')[0]
        self.protein_id2name = self.dataloader.get_dict(type='protein')[0]
        self.biof_id2name = self.dataloader.get_dict(type='biological_function')[0]
        
        protein_to_protein = pd.read_csv(abs_raw_path / '3_protein_to_protein.tsv', sep='\t')
        protein_to_protein['node_1'] = protein_to_protein['node_1'].map(str)
        protein_to_protein['node_2'] = protein_to_protein['node_2'].map(str)
        
        protein_to_biof = pd.read_csv(abs_raw_path / '4_protein_to_biological_function.tsv', sep='\t')
        protein_to_biof['node_1'] = protein_to_biof['node_1'].map(str)
        
        biof_to_biof = pd.read_csv(abs_raw_path / '5_biological_function_to_biological_function.tsv', sep='\t')
        
        indication_to_protein = pd.read_csv(abs_raw_path / '2_indication_to_protein.tsv', sep='\t')
        indication_to_protein['node_2'] = indication_to_protein['node_2'].map(str)
        
        drug_to_protein = pd.read_csv(abs_raw_path / '1_drug_to_protein.tsv', sep='\t')
        drug_to_protein['node_2'] = drug_to_protein['node_2'].map(str)
        
        self.graph = pd.concat([protein_to_protein,
                                      protein_to_biof,
                                      biof_to_biof,
                                      indication_to_protein,
                                      drug_to_protein],
                                     axis=0)
        
    def get_prof(self, id_):
        '''
        id_: ID of drug or indication
        
        returns a diffusion profile vector of id_
        '''
        filename = id_ + '_p_visit_array.npy'
        if filename in self.filenames:
            path = os.path.join(self.path, filename)
            prof = np.load(path)
            return prof
        else:
            print("There is no diffusion profile for "+str(id_))
            return None
        
    def get_type_profs(self, type='drug') -> List[Tuple]:
        '''
        type: 'drug' or 'indication'
        
        returns a list of tuples - (id_, name, diffusion profile vector)
        '''
        
        type_id2name = self.dataloader.get_dict(type)
        type_profs = []
        for filename in self.filenames:
            id_ = filename.split('_')[0]
            if id_ in type_id2name.keys():
                type_profs.append( (id_, type_id2name[id_], self.get_prof(id_)) )
        return type_profs
    
    def rank_profs(self, src_prof, type_profs):
        '''
        source_prof: a diffusion profile vector of a source node
        type_profs: a list of tuples - (id_, name, diffusion profile vector)
    
        Ranking by correlation distance
    
        returns a sorted(key=similarity score) list of tupels - (id_, name, diffusion profile vector)
        '''
        list_rank = []
        for tuple_ in type_profs:
            prof = tuple_[2]
            similarity = -1 * scipy.spatial.distance.correlation(prof, src_prof)
            list_rank.append((tuple_[0], tuple_[1], similarity))
        list_rank.sort(key=lambda x: x[2], reverse=True)
        return list_rank
    
    def rank_top_k_nodes(self, prof, k=10, verbose=False) -> List[Tuple]:
        '''
        ranks top protein or biological function nodes.
        
        returns a list of tuples - (top k protein or biological function node ID, its name)
        '''
        prof_dropped = copy.deepcopy(prof)
        prof_dropped[9798+17660:] = 0.0
        
        if verbose:
            print("Sum of diffusion vector elements after dropping drug and indication nodes: ", prof_dropped.sum())
        
        top_k_idxs = np.argsort(-prof_dropped)[:k] # descending order
        
        top_k = []
        for idx in top_k_idxs:
            id_ = self.idx2node[idx]
            if id_ in self.protein_id2name:
                name = self.protein_id2name[id_]
            elif id_ in self.biof_id2name:
                name = self.biof_id2name[id_]
            else:
                print("Wrong ID:", idx)
                name = 'dummy'
            top_k.append((id_, name))

        return top_k
    
    def write_top_k_nodes(self, src_id, k=10):
        prof = self.get_prof(src_id)
        top_k = self.rank_top_k_nodes(prof, k=k, verbose=False)
        if src_id in self.drug_id2name:
            node_name = self.drug_id2name[src_id]
        elif src_id in self.indication_id2name:
            node_name = self.indication_id2name[src_id]
        
        filename = str(src_id) + '.txt'
        with open(filename, 'w') as f:
            for x in top_k:
                f.write((str(x[0]) + '\t' + str(x[1])))
                f.write('\n')
        
    def get_graph_tsv(self, node_1_id, node_2_id, k=10):
        list_nodes = []
        if node_1_id in self.drug_id2name:
            node_1_name = self.drug_id2name[node_1_id]
            list_nodes.append((node_1_id, node_1_name))
        elif node_1_id in self.indication_id2name:
            node_1_name = self.indication_id2name[node_1_id]
            list_nodes.append((node_1_id, node_1_name))
        
        if node_2_id in self.drug_id2name:
            node_2_name = self.drug_id2name[node_2_id]
            list_nodes.append((node_2_id, node_2_name))
        elif node_2_id in self.indication_id2name:
            node_2_name = self.indication_id2name[node_2_id]
            list_nodes.append((node_2_id, node_2_name))

        node_1_prof = self.get_prof(node_1_id)
        node_2_prof = self.get_prof(node_2_id)

        node_1_top_k_nodes = self.rank_top_k_nodes(node_1_prof, k=k)
        node_2_top_k_nodes = self.rank_top_k_nodes(node_2_prof, k=k)

        for tuple_ in node_1_top_k_nodes:
            list_nodes.append(tuple_)
        for tuple_ in node_2_top_k_nodes:
            list_nodes.append(tuple_)

        list_edges = []
        for i in range(len(list_nodes)):
            for j in range(i+1, len(list_nodes)):
                tuple_1 = list_nodes[i]
                tuple_2 = list_nodes[j]
                id_1 = tuple_1[0]
                id_2 = tuple_2[0]

                try:
                    edge = self.graph[((self.graph['node_1'] == id_1) & (self.graph['node_2'] == id_2)) | ((self.graph['node_1'] == id_2) & (self.graph['node_2'] == id_1))].values[0].tolist()
                    list_edges.append(edge)
                except:
                    pass
        df = pd.DataFrame(list_edges, columns=['node_1', 'node_2', 'node_1_type', 'node_2_type', 'node_1_name', 'node_2_name'])
        df.to_csv(node_1_name + '_and_' + node_2_name + '.tsv', index=False, header=True, sep='\t')
        return df