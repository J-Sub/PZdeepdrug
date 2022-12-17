import pickle
from load_msi_data import LoadData

import torch
import numpy as np
import pandas as pd
import os
import random
from pathlib import Path

from torch.utils.data import Dataset, DataLoader

class CombinationDataset(Dataset):
    def __init__(self, database='DCDB', neg_ratio=1, duplicate=False, use_ddi=False, ddi_dataset=None, seed=42, transform=None):
        '''
        args
            - database: str, default='C_DCDB' ['C_DCDB', 'DCDB', 'DC_combined']
            - neg_ratio: int, default=1
            - duplicate: bool, default=False (if True, duplicate each samples) -> [a, b] & [b, a]
            - use_ddi: bool, default=False (if True, use ddi dataset)
            - ddi_dataset: str, default=None (if use_ddi is True, choose ddi dataset) ['DB', 'TWOSIDES']
            - seed: int, default=42
        '''
        if (database != 'DCDB') & (database != 'C_DCDB') & (database != 'DC_combined'):
            raise ValueError('database must be DCDB or C_DCDB or DC_combined')
        if neg_ratio < 1:
            raise ValueError('neg_ratio must be greater than 1')
        if use_ddi:
            if (ddi_dataset != 'DB') & (ddi_dataset != 'TWOSIDES'):
                raise ValueError('ddi_dataset must be DB or TWOSIDES')
        
        self.database = database
        self.neg_ratio = neg_ratio
        self.transform = transform
        self.duplicate = duplicate
        self.use_ddi = use_ddi
        self.ddi_dataset = ddi_dataset
        self.seed = seed

        self.data_path = Path('data/processed')/f'{database}_neg{neg_ratio}_dup{int(duplicate)}_ddi{int(use_ddi)}_{ddi_dataset}_seed{seed}.pt'
        if self.data_path.exists():
            print(f'{self.data_path} already exists in processed/ directory.')
        else:
            self._process()
        print(f'Loading dataset...{self.data_path}')
        print('Dictionary of {train, valid, test} dataset is loaded.')
        self.data = torch.load(self.data_path)
        
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]
    
    def _process(self):
        print('Processing dataset...')
        dataset_list = self._create_dataset()
        train_size = int(len(dataset_list) * 0.8)
        valid_size = int(len(dataset_list) * 0.1)
        test_size = len(dataset_list) - train_size - valid_size
        torch.manual_seed(0)
        train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(dataset_list, [train_size, valid_size, test_size])
        dataset_dict = {'train': train_dataset, 'valid': valid_dataset, 'test': test_dataset
        }

        print(f'Saving dataset...')
        torch.save(dataset_dict, self.data_path)
    
    def _create_dataset(self):
        dataloader = LoadData()
        # Get embedding
        with open('data/embedding/embeddings_node2vec_msi_seed0.pkl', 'rb') as f:
            embedding_dict = pickle.load(f)
        # drug dictionary
        drug_id2name, drug_name2id = dataloader.get_dict(type='drug')
        # Prepare positive labels
        pos_df = pd.read_csv(f'data/labels/{self.database}_msi.tsv', sep='\t')

        dataset_list = []
        # positive samples
        for i in range(len(pos_df)):
            drug1_id = pos_df.iloc[i, 0]
            drug2_id = pos_df.iloc[i, 1]
            comb_embedding = np.concatenate([embedding_dict[drug1_id], embedding_dict[drug2_id]])
            dataset_list.append([torch.tensor(comb_embedding, dtype=torch.float), torch.tensor(1, dtype=torch.long)])
            if self.duplicate:
                comb_embedding2 = np.concatenate([embedding_dict[drug2_id], embedding_dict[drug1_id]])
                dataset_list.append([torch.tensor(comb_embedding2, dtype=torch.float), torch.tensor(1, dtype=torch.long)])
                
        # negative samples
        if self.use_ddi:
            neg_df = pd.read_csv(f'data/labels/{self.ddi_dataset}_DDI_msi.tsv', sep='\t')
            if len(neg_df) < len(pos_df) * self.neg_ratio:
                raise ValueError('Not enough negative samples')
            
            neg_df = neg_df.sample(n=len(pos_df) * self.neg_ratio, random_state=self.seed)
            for i in range(len(neg_df)):
                drug1_id = neg_df.iloc[i, 0]
                drug2_id = neg_df.iloc[i, 1]
                comb_embedding = np.concatenate([embedding_dict[drug1_id], embedding_dict[drug2_id]])
                dataset_list.append([torch.tensor(comb_embedding, dtype=torch.float), torch.tensor(0, dtype=torch.long)])
                if self.duplicate:
                    comb_embedding2 = np.concatenate([embedding_dict[drug2_id], embedding_dict[drug1_id]])
                    dataset_list.append([torch.tensor(comb_embedding2, dtype=torch.float), torch.tensor(0, dtype=torch.long)])
        else:
            count = 0
            while count < len(pos_df) * self.neg_ratio:
            # while len(dataset_list) < len(pos_df) * (1 + self.neg_ratio):
                drug1_id = random.choice(list(drug_id2name.keys()))
                drug2_id = random.choice(list(drug_id2name.keys()))
                if drug1_id == drug2_id:
                    continue
                if ((pos_df['drug_1'] == drug1_id) & (pos_df['drug_2'] == drug2_id)).any():
                    continue
                if ((pos_df['drug_1'] == drug2_id) & (pos_df['drug_2'] == drug1_id)).any():
                    continue
                comb_embedding = np.concatenate([embedding_dict[drug1_id], embedding_dict[drug2_id]])
                dataset_list.append([torch.tensor(comb_embedding, dtype=torch.float), torch.tensor(0, dtype=torch.long)])
                if self.duplicate:
                    comb_embedding2 = np.concatenate([embedding_dict[drug2_id], embedding_dict[drug1_id]])
                    dataset_list.append([torch.tensor(comb_embedding2, dtype=torch.float), torch.tensor(0, dtype=torch.long)])
                count += 1
        return dataset_list