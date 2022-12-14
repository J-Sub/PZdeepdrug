import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CombNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, comb_type='cat', dropout=0.1):
        super(CombNet, self).__init__()
        self.input_dim = input_dim # dimension of concatenated drug embeddings
        if (comb_type != 'cat') & (comb_type != 'sum') & (comb_type != 'diff') & (comb_type != 'sumdiff') & (comb_type != 'cosine') & (comb_type != "euclidean"):
            raise ValueError('comb_type must be cat, sum, diff, sumdiff, or cosine.')
        self.comb_type = comb_type
        self.lt = nn.Sequential(
            nn.Linear(input_dim // 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        if comb_type == 'cosine':
            # # self.fc = nn.Sequential(
            # #     nn.Linear(1, output_dim)
            # # )
            # raise NotImplementedError('cosine similarity is not implemented yet.')
            self.metric = nn.CosineSimilarity(dim=1, eps=1e-6)
        else:
            dual_dim = hidden_dim if (comb_type == 'sum' or comb_type == 'diff') else hidden_dim * 2
            self.fc = nn.Sequential(
                nn.Linear(dual_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                # nn.Linear(hidden_dim, hidden_dim),
                # nn.BatchNorm1d(hidden_dim),
                # nn.ReLU(),
                # nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim) # for BCEWithLogitsLoss
            )
            # self.fc = nn.Sequential(
            #     nn.Linear(dual_dim, output_dim)
            # )


    def forward(self, data):
        drug1, drug2 = data[:, :self.input_dim//2], data[:, self.input_dim//2:] # drug1과 drug2를 분리 # (batch_size * (input_dim // 2))
        drug1, drug2 = self.lt(drug1), self.lt(drug2)
        if self.comb_type == 'cosine':
            drug1 = F.normalize(drug1, dim=1)
            drug2 = F.normalize(drug2, dim=1)
            # define cosine similarity
            cos_sim = self.metric(drug1, drug2).unsqueeze(1) # (batch_size, 1), scale: -1 ~ 1
            # scale cos_sim to 0 ~ 1
            angular_dist = torch.arccos(cos_sim) / np.pi # (batch_size, 1), scale: 0 ~ 1
            return angular_dist
        else:
            if self.comb_type == 'cat': # [(drug1), (drug2)]로 concat
                comb = torch.cat([drug1, drug2], dim=1) # (batch_size, hidden_dim * 2)
            elif self.comb_type == 'sum': # [(drug1) + (drug2)]
                comb = drug1 + drug2 # (batch_size, hidden_dim)
            elif self.comb_type == 'diff': # [(drug1) - (drug2)]
                comb = torch.abs(drug1 - drug2) # (batch_size, hidden_dim)
            elif self.comb_type == 'sumdiff': # [(drug1) + (drug2), (drug1) - (drug2)]로 concat
                comb = torch.cat([drug1 + drug2, torch.abs(drug1 - drug2)], dim=1) # (batch_size, hidden_dim * 2)
            return self.fc(comb) # (batch_size, output_dim)

class CombNetSupCon(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, contrastive_dim, dropout=0.1):
        super(CombNetSupCon, self).__init__()
        
        self.input_dim = input_dim
        self.tr = nn.Sequential(
            nn.Linear(input_dim // 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            # nn.Dropout(dropout),
        )
        self.contrastive_hidden = nn.Sequential(
            nn.Linear(hidden_dim * 2, contrastive_dim),
            nn.BatchNorm1d(contrastive_dim),
            nn.ReLU(),
        )
        self.contrastive_out = nn.Linear(contrastive_dim, contrastive_dim)

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.BatchNorm1d(hidden_dim),
            # nn.ReLU(),
            # nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim) # for BCEWithLogitsLoss
        )    
        # self.fc = nn.Sequential(
        #     nn.Linear(hidden_dim * 2, output_dim) # for BCEWithLogitsLoss
        # )
    
    def freeze_projection(self):
        self.tr.requires_grad_(False)

    def _forward_impl(self, data):
        drug1, drug2 = data[:, :self.input_dim//2], data[:, self.input_dim//2:]
        drug1, drug2 = self.tr(drug1), self.tr(drug2)
        comb = torch.cat([drug1, drug2], dim=1)
        out = F.normalize(comb, dim=1)
        return out

    def forward_contrastive(self, data):
        x = self._forward_impl(data)
        x = self.contrastive_hidden(x)
        x = self.contrastive_out(x)
        # normalize to unit hypersphere
        x = F.normalize(x, dim=1)
        return x

    def forward(self, data):
        x = self._forward_impl(data)
        return self.fc(x)

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

class CombNetSupCon(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, contrastive_dim, dropout=0.1):
        super(CombNetSupCon, self).__init__()

        self.input_dim = input_dim
        self.tr = nn.Sequential(
            nn.Linear(input_dim // 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )

        self.contrastive = nn.Sequential(
            nn.Linear(hidden_dim, contrastive_dim),
            nn.BatchNorm1d(contrastive_dim),
            nn.ReLU(),
            nn.Linear(contrastive_dim, contrastive_dim)
        )

        self.fc == nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim) # for BCEWithLogitsLoss
        )

    def freeze_projection(self):
        self.tr.requires_grad_(False)
    
    def _forward_impl(self, data):
        drug1, drug2 = data[:, :self.input_dim//2], data[:, self.input_dim//2:] # (batch_size, input_dim // 2)
        # drug1, drug2 = F.normalize(self.tr(drug1), dim=1), F.normalize(self.tr(drug2), dim=1)
        return drug1, drug2 # (batch_size, hidden_dim) * 2
    
    def forward_contrastive(self, data):
        drug1, drug2 = self._forward_impl(data)
        drug1, drug2 = F.normalize(self.contrastive(drug1), dim=1), F.normalize(self.contrastive(drug2), dim=1)
        return drug1, drug2 # (batch_size, contrastive_dim) * 2
    
    def forward(self, data):
        drug1, drug2 = self._forward_impl(data)
        comb = torch.cat([drug1, drug2], dim=1)
        return self.fc(comb) # (batch_size, output_dim)