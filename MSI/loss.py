import torch
import torch.nn as nn

from math import log

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, projections, targets):
        '''
        Args:
            projections: torch.Tensor, shape [batch_size, projection_dim]
            targets: torch.Tensor, shape [batch_size]

        Returns:
            loss: torch.Tensor, shape [1]
        '''

        device = torch.device('cuda') if projections.is_cuda else torch.device('cpu')
        dot_product_tempered = torch.mm(projections, projections.T) / self.temperature # [batch_size, batch_size]

        # minus max for numerical stability with exponential
        exp_dot_tempered = (
            torch.exp(dot_product_tempered - torch.max(dot_product_tempered, dim=1, keepdim=True)[0]) + 1e-5
        ) # [batch_size, batch_size]

        mask_similar_class = (targets.unsqueeze(1).repeat(1, targets.shape[0]) == targets).to(device) # [batch_size, batch_size]
        mask_anchor_out = (1 - torch.eye(targets.shape[0])).to(device) # [batch_size, batch_size] (1 - I)
        mask_combined = mask_similar_class * mask_anchor_out # [batch_size, batch_size]
        cardinality_per_samples = torch.sum(mask_combined, dim=1) # [batch_size]

        log_prob = - torch.log(exp_dot_tempered / torch.sum(exp_dot_tempered * mask_anchor_out, dim=1, keepdim=True)) # [batch_size, batch_size]
        sup_con_loss_per_sample = torch.sum(log_prob * mask_combined, dim=1) # [batch_size]
        sup_con_loss = torch.mean(sup_con_loss_per_sample) # [1]

        return sup_con_loss
        