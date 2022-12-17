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
    parser.add_argument('--database', type=str, default='C_DCDB')
    parser.add_argument('--neg_ratio', type=int, default=1)
    parser.add_argument('--duplicate', type=bool, default=False)
    parser.add_argument('--use_ddi', type=bool, default=False)
    parser.add_argument('--ddi_dataset', type=str, default=None)
    parser.add_argument('--comb_type', type=str, default='cat') # cat, sum, diff, sumdiff
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--train_mode', type=str, default='cross_entropy')
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



def train_sup_con(model, device, train_loader, criterion, optimizer):
    model.train()
    train_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        projections = model.forward_contrastive(data)
        loss = criterion(projections, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    
    return train_loss / (batch_idx + 1)

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
    wandb.init(project="PharmGen_drug_comb")
    args = parse_args()
    wandb.config.update(args)
    print(args)

    seed_everything(args.seed)

    dataset = CombinationDataset(database=args.database, neg_ratio=args.neg_ratio, duplicate=args.duplicate, use_ddi=args.use_ddi, ddi_dataset=args.ddi_dataset, seed=args.seed)
    print(len(dataset))

    train_size = int(0.8 * len(dataset))
    valid_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - valid_size
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    input_dim = dataset[0][0].shape[0]
    hidden_dim = input_dim
    output_dim = 1

    EPOCHS = args.epochs
    LR = args.lr
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')

    if args.train_mode == 'cross_entropy':
        print('Train with cross entropy')
        model = CombNet(input_dim, hidden_dim, output_dim, comb_type=args.comb_type)
        model.to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=args.weight_decay)

        best_valid_loss = float('inf')
        for epoch in range(EPOCHS):
            train_loss, train_scores = train_cross_entropy(model, device, train_loader, criterion, optimizer,
                                                           metric_list=[accuracy_score, roc_auc_score, f1_score, average_precision_score])
            valid_loss, valid_scores = evaluate(model, device, valid_loader, criterion, metric_list=[accuracy_score, roc_auc_score, f1_score, average_precision_score])
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), 'checkpoint.pt')
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
            })
            print(f'Epoch {epoch+1:03d}: | Train Loss: {train_loss:.4f} | Train Acc: {train_scores[0]*100:.2f}% | Train AUROC: {train_scores[1]:.2f} | Train F1: {train_scores[2]:.4f} | Train AUPRC: {train_scores[3]:.2f} || Val. Loss: {valid_loss:.4f} | Val. Acc: {valid_scores[0]*100:.2f}% | Val. AUROC: {valid_scores[1]:.2f} | Val. F1: {valid_scores[2]:.4f} | Val. AUPRC: {valid_scores[3]:.2f}')
        
        test_loss, test_scores = evaluate(model, device, test_loader, criterion, metric_list=[accuracy_score, roc_auc_score, f1_score, average_precision_score], checkpoint='checkpoint.pt')
        print(f'Test Loss: {test_loss:.4f} | Test Acc: {test_scores[0]*100:.2f}% | Test AUROC: {test_scores[1]:.2f} | Test F1: {test_scores[2]:.4f} | Test AUPRC: {test_scores[3]:.2f}')
    
    elif args.train_mode == 'sup_con':
        print('Supervised Contrastive pretraining...')
        pass


if __name__ == '__main__':
    main()