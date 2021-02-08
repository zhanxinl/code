import torch
from torch import nn
import torch.nn.functional as F

bcew_loss = torch.nn.BCEWithLogitsLoss(reduction="mean")  # D_loss_temp
mse_loss = torch.nn.MSELoss(reduction="mean")
bce_loss=torch.nn.BCELoss(reduction="mean")


def discriminator_loss(D_logit,M):
    D_loss =bcew_loss(D_logit,M)
    return D_loss

def generator_loss(New_X, M, G_sample, D_logit):
    adversarial_G_loss =-torch.mean((1 - M) * (torch.sigmoid(D_logit)+1e-8).log())/(1-M).mean()
    reconstruct_loss=mse_loss(M*New_X, M*G_sample)/ torch.mean(M)
    return  adversarial_G_loss,reconstruct_loss

def test_loss(X_mb, M_mb,G_prob):
    mse_test_loss = mse_loss((1-M_mb)*X_mb, (1-M_mb)*G_prob) / torch.mean(1-M_mb)
    return mse_test_loss
