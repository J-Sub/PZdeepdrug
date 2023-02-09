import torch
import numpy as np
import pandas as pd
import os
import random
from pathlib import Path

import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchmetrics import AUROC, Accuracy, Precision, Recall
from torchmetrics.classification import BinaryAUROC, BinaryF1Score

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, average_precision_score

import pickle
from load_msi_data import LoadData
from model import CombNet, CombNetSupCon
from dataset import CombinationDataset
from loss import SupConLoss

import argparse

import wandb

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--database', type=str, default='C_DCDB', choices=['C_DCDB', 'DCDB', 'DC_combined'])
    parser.add_argument('--embeddingf', type=str, default='node2vec', choices=['node2vec', 'edge2vec', 'res2vec_homo', 'res2vec_hetero', 'DREAMwalk'])
    parser.add_argument('--neg_dataset', type=str, default='random', choices=['random', 'TWOSIDES'])
    parser.add_argument('--neg_ratio', type=int, default=1)
    parser.add_argument('--duplicate', type=bool, default=False)
    parser.add_argument('--comb_type', type=str, default='cat', choices=['cat', 'sum', 'diff', 'sumdiff', 'cosine'])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=int, default=0)
#     parser.add_argument('--run_name', type=str, default='test_run')
    parser.add_argument('--group', type=str, default=None)
    parser.add_argument('--ckpt_name', type=str, default='checkpoint.pt')
    args = parser.parse_args()
    return args

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0, path='checkpoint.pt'):
        '''
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
        '''
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
    
    def __call__(self, val_loss, model):
        
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
    
    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

# def train_sup_con(model, device, train_loader, criterion, optimizer):
#     model.train()
#     train_loss = 0

#     for batch_idx, (data, target) in enumerate(train_loader):
#         data, target = data.to(device), target.to(device)
#         optimizer.zero_grad()
#         projections = model.forward_contrastive(data)
#         loss = criterion(projections, target)
#         loss.backward()
#         optimizer.step()

#         train_loss += loss.item()
    
#     return train_loss / (batch_idx + 1)

def train_cross_entropy(model, device, train_loader, criterion, optimizer, metric_list=[accuracy_score]):

    # train
    model.train()
    train_loss = 0

    target_list = []
    pred_list = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.float().to(device)
        optimizer.zero_grad()
        output = model(data).view(-1) # z
        # print(output)
        loss = criterion(output, target) # z, y
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        pred_list.append(torch.sigmoid(output).detach().cpu().numpy())
        target_list.append(target.long().detach().cpu().numpy())
    
    # metric
    scores = []
    for metric in metric_list:
        if (metric == roc_auc_score) or (metric == average_precision_score):
            scores.append(metric(np.concatenate(target_list), np.concatenate(pred_list)))
        else: # accuracy_score, f1_score, precision_score, recall_score
            scores.append(metric(np.concatenate(target_list), np.concatenate(pred_list).round()))
    
    return train_loss / (batch_idx + 1), scores

def evaluate(model, device, loader, criterion, metric_list=[accuracy_score], checkpoint=None):
    # evaluate
    if checkpoint is not None:
        model.load_state_dict(torch.load(checkpoint))
    model.eval()
    eval_loss = 0

    target_list = []
    pred_list = []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.to(device), target.float().to(device)
            output = model(data).view(-1)
            eval_loss += criterion(output, target).item()
            pred_list.append(torch.sigmoid(output).detach().cpu().numpy())
            target_list.append(target.long().detach().cpu().numpy())

    scores = []
    for metric in metric_list:
        if (metric == roc_auc_score) or (metric == average_precision_score):
            scores.append(metric(np.concatenate(target_list), np.concatenate(pred_list)))
        else: # accuracy_score, f1_score, precision_score, recall_score
            scores.append(metric(np.concatenate(target_list), np.concatenate(pred_list).round()))
    return eval_loss / (batch_idx + 1), scores


def main():
    args = parse_args()
    group = f"{args.database}_{args.embeddingf}_neg({args.neg_dataset}_{args.neg_ratio})_comb({args.comb_type})" #
    wandb.init(project="PharmGen_drug_comb", group=group, entity='pzdeepdrug')
    wandb.config.update(args)
    wandb.run.name = f"{args.database}_{args.embeddingf}_neg({args.neg_dataset}_{args.neg_ratio})_comb({args.comb_type})_seed{args.seed}"
    wandb.run.save()
    print(args)

    seed_everything(args.seed)

    dataset = CombinationDataset(database=args.database, embeddingf=args.embeddingf, neg_ratio=args.neg_ratio, duplicate=args.duplicate, neg_dataset=args.neg_dataset, seed=args.seed)
    train_dataset, valid_dataset, test_dataset = dataset['train'], dataset['valid'], dataset['test']
    print(len(train_dataset), len(valid_dataset), len(test_dataset))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    input_dim = train_dataset[0][0].shape[0]
    hidden_dim = input_dim
    output_dim = 1

    EPOCHS = args.epochs
    LR = args.lr
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')

    print('Train with cross entropy')
    model = CombNet(input_dim, hidden_dim, output_dim, comb_type=args.comb_type)
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=args.weight_decay)

    best_valid_loss = float('inf')

    early_stopping = EarlyStopping(patience=10, verbose=True, path=args.ckpt_name)

    for epoch in range(EPOCHS):
        train_loss, train_scores = train_cross_entropy(model, device, train_loader, criterion, optimizer,
                                                       metric_list=[accuracy_score, roc_auc_score, f1_score, average_precision_score, precision_score, recall_score])
        valid_loss, valid_scores = evaluate(model, device, valid_loader, criterion, metric_list=[accuracy_score, roc_auc_score, f1_score, average_precision_score, precision_score, recall_score])
        # if valid_loss < best_valid_loss:
        #     best_valid_loss = valid_loss
        #     torch.save(model.state_dict(), 'checkpoint.pt')
        
        wandb.log({
            'train_loss': train_loss,
            'valid_loss': valid_loss,
            'train_accuracy': train_scores[0],
            'valid_accuracy': valid_scores[0],
            'train_auroc': train_scores[1],
            'valid_auroc': valid_scores[1],
            'train_f1': train_scores[2],
            'valid_f1': valid_scores[2],
            'train_auprc': train_scores[3],
            'valid_auprc': valid_scores[3],
            'train_precision': train_scores[4],
            'valid_precision': valid_scores[4],
            'train_recall': train_scores[5],
            'valid_recall': valid_scores[5],
        })
        print(f'Epoch {epoch+1:03d}: | Train Loss: {train_loss:.4f} | Train Acc: {train_scores[0]*100:.2f}% | Train Precision: {train_scores[4]:.4f} | Train Recall: {train_scores[5]:.4f} || Valid Loss: {valid_loss:.4f} | Valid Acc: {valid_scores[0]*100:.2f}% | Valid Precision: {valid_scores[4]:.4f} | Valid Recall: {valid_scores[5]:.4f}')
        early_stopping(valid_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    test_loss, test_scores = evaluate(model, device, test_loader, criterion, metric_list=[accuracy_score, roc_auc_score, f1_score, average_precision_score, precision_score, recall_score], checkpoint=args.ckpt_name)
    wandb.log({
        'test_loss': test_loss,
        'test_accuracy': test_scores[0],
        'test_auroc': test_scores[1],
        'test_f1': test_scores[2],
        'test_auprc': test_scores[3],
        'test_precision': test_scores[4],
        'test_recall': test_scores[5],
    })
    print(f'Test Loss: {test_loss:.4f} | Test Acc: {test_scores[0]*100:.2f}% | Test Precision: {test_scores[4]:.4f} | Test Recall: {test_scores[5]:.4f}')
    


if __name__ == '__main__':
    main()