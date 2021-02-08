import torch
from torch import nn
import torch.nn.functional as F


class NetG(nn.Module):
    def __init__(self,Dim,H_Dim, dropout=0.1):
        super(NetG,self).__init__()
        main = nn.Sequential(
             # input is mix of orginal data and noise, and mask  the dimension =dataset dimension*2
              # state size. (opt.dim*2)
             nn.Linear(Dim*2, H_Dim*2),
             nn.BatchNorm1d(H_Dim*2),
             nn.ReLU(),
             nn.Dropout(dropout),
              # state size. (opt.dim)
             nn.Linear(H_Dim*2,H_Dim),
             nn.BatchNorm1d(H_Dim),
             nn.ReLU(),
             nn.Dropout(dropout),
            # state size. (dataset dim)
             nn.Linear(H_Dim,Dim),
             nn.Sigmoid()
         )
        self.main = main
        self._initialize()

    def _initialize(self):
        for layer in self.main:
            if type(layer) in [nn.Linear]:
                torch.nn.init.xavier_normal_(layer.weight) 
            
    def forward(self, inp, m):
#         inp = m * x + (1-m) * z
        inp = torch.cat((inp, m), dim=1)
        G_sample = self.main(inp) 
        return G_sample

class NetC(nn.Module):
    def __init__(self,Dim,n_class,H_Dim, dropout=0.1):
        super(NetC,self).__init__()
        main = nn.Sequential(
             #  imputed data and orginal data the dimension =dataset dimension
             nn.Linear(Dim, H_Dim*2),
             nn.BatchNorm1d(H_Dim*2),
             nn.ReLU(),
             nn.Dropout(dropout),
              # state size. (opt.dim*2)
             nn.Linear(H_Dim*2,H_Dim),
             nn.BatchNorm1d(H_Dim),
             nn.ReLU(),
             nn.Dropout(dropout),
              # state size. (opt.dim)
             nn.Linear(H_Dim,n_class),
              # state size. (n_class)
             nn.Sigmoid()
         )
        self.main = main
        self._initialize()
    
    def _initialize(self):
        for layer in self.main:
            if type(layer) in [nn.Linear]:
                torch.nn.init.xavier_normal_(layer.weight) 
    
    def forward(self, inp):
#         inp = m * x + (1-m) * G
        C_logits = self.main(inp) 
        return C_logits
    
    
class NetD_noC(nn.Module):

    def __init__(self,Dim,H_Dim, dropout=0.1):
        super(NetD_noC, self).__init__()
        main = nn.Sequential(
             #  imputed data and orginal data the dimension =dataset dimension
             nn.BatchNorm1d(Dim*2),
             nn.Linear(Dim*2, H_Dim*2),
             nn.BatchNorm1d(H_Dim*2),
             nn.ReLU(),
             nn.Dropout(dropout),
              # state size. (opt.dim*2)
             nn.Linear(H_Dim*2,H_Dim),
             nn.BatchNorm1d(H_Dim),
             nn.ReLU(),
             nn.Dropout(dropout),
            # state size. (dataset dim)
             nn.Linear(H_Dim,Dim),
             nn.Sigmoid()
         )
        self.main = main
        
        self._initialize()
    def _initialize(self):
        for layer in self.main:
            if type(layer) in [nn.Linear]:
                torch.nn.init.xavier_normal_(layer.weight) 

    def forward(self,inp,h): 
        # Get output
        inp = torch.cat((inp, h), dim=1)
        D_logits = self.main(inp)
        return D_logits

class NetD(nn.Module):
    def __init__(self,Dim,H_Dim,n_class, dropout=0.1):
        super(NetD, self).__init__()
        main = nn.Sequential(
             #  imputed data and orginal data the dimension =dataset dimension
             nn.BatchNorm1d(Dim*3),
             nn.Linear(Dim*3, H_Dim*2),
             nn.BatchNorm1d(H_Dim*2),
             nn.ReLU(),
             nn.Dropout(dropout),
              # state size. (opt.dim*2)
             nn.Linear(H_Dim*2,H_Dim),
             nn.BatchNorm1d(H_Dim),
             nn.ReLU(),
             nn.Dropout(dropout),
            # state size. (dataset dim)
             nn.Linear(H_Dim,Dim),
             nn.Sigmoid()
         )
        self.main = main
        self.y_layer=nn.Linear(n_class,Dim)
        
        self._initialize()
        torch.nn.init.xavier_normal_(self.y_layer.weight) 
    
    def _initialize(self):
        for layer in self.main:
            if type(layer) in [nn.Linear]:
#                 print(layer)
                torch.nn.init.xavier_normal_(layer.weight) 

    def forward(self,inp,h,y): 
        # Get output
        y=self.y_layer(y)
        inp = torch.cat((inp, h), dim=1)
        inp = torch.cat((inp, y), dim=1)
        D_logits = self.main(inp)
        return D_logits