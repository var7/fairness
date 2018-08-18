import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class AdvDemographicParityLoss(nn.Module):
    def __init__(self, p=1):
        super(AdvDemographicParityLoss, self).__init__()
        if p == 1:
            self.distance = nn.L1Loss()
        elif p == 2:
            self.distance = nn.MSELoss()
        else:
            print('P has to be either 1 or 2')
        
    def forward(self, predicted_sensitive, sensitive):
        if not (sensitive.shape[0] == predicted_sensitive.shape[0]):
            raise ValueError("Target size ({}) must be the same as predicted_sensitive size ({})".format(target.size(), predicted_sensitive.size()))
        
        if isinstance(sensitive, np.ndarray):
            sensitive = torch.from_numpy(sensitive)
            sensitive = sensitive.float()
        
#         predicted_sensitive = predicted_sensitive.squeeze(dim=1)
        
        a0_mask = (sensitive == 0)
        a1_mask = (sensitive == 1)
        
        predicted_sensitive_group0 = predicted_sensitive[a0_mask]
        predicted_sensitive_group1 = predicted_sensitive[a1_mask]
        
        target_group0 = sensitive[a0_mask]
        target_group1 = sensitive[a1_mask]
        
        loss = 1 - (self.distance(predicted_sensitive_group0, target_group0) + self.distance(predicted_sensitive_group1, target_group1))
        return loss
        
class AdvEqOddsLoss(nn.Module):
    def __init__(self, p=1):
        super(AdvEqOddsLoss, self).__init__()
        if p == 1:
            self.distance = nn.L1Loss()
        elif p == 2:
            self.distance = nn.MSELoss()
        else:
            print('P has to be either 1 or 2')
        
    def forward(self, predicted_sensitive, sensitive, targets):
        if not (targets.shape[0] == predicted_sensitive.shape[0]):
            raise ValueError("Target size ({}) must be the same as predicted_sensitive size ({})".format(target.size(), predicted_sensitive.size()))
        
        if isinstance(targets, np.ndarray):
            targets = torch.from_numpy(targets)
            targets = targets.float()
            
        if isinstance(sensitive, np.ndarray):
            sensitive = torch.from_numpy(sensitive)
            sensitive = sensitive.float()
        
        
#         predicted_sensitive = predicted_sensitive.squeeze(dim=1)
        
        a00_mask = ((targets == 0).byte() & (sensitive == 0).byte())
        a01_mask = ((targets == 1).byte() & (sensitive == 0).byte())
        a10_mask = ((targets == 0).byte() & (sensitive == 1).byte())
        a11_mask = ((targets == 1).byte() & (sensitive == 1).byte())

        predicted_sensitive_group00 = predicted_sensitive[a00_mask]
        predicted_sensitive_group01 = predicted_sensitive[a01_mask]
        predicted_sensitive_group10 = predicted_sensitive[a10_mask]
        predicted_sensitive_group11 = predicted_sensitive[a11_mask]

        true_sensitive_group00 = sensitive[a00_mask]
        true_sensitive_group01 = sensitive[a01_mask]
        true_sensitive_group10 = sensitive[a10_mask]
        true_sensitive_group11 = sensitive[a11_mask]
         
        distance00 = 0 if torch.isnan(self.distance(predicted_sensitive_group00, true_sensitive_group00)) else self.distance(predicted_sensitive_group00, true_sensitive_group00)
        distance01 = 0 if torch.isnan(self.distance(predicted_sensitive_group01, true_sensitive_group01)) else self.distance(predicted_sensitive_group01, true_sensitive_group01)
        distance10 = 0 if torch.isnan(self.distance(predicted_sensitive_group10, true_sensitive_group10)) else self.distance(predicted_sensitive_group10, true_sensitive_group10)
        distance11 = 0 if torch.isnan(self.distance(predicted_sensitive_group11, true_sensitive_group11)) else self.distance(predicted_sensitive_group11, true_sensitive_group11)
        
        loss = 2 - (distance00 + distance01 + distance10 + distance11)
           
        return loss