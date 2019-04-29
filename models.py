import torch
from torch import nn
from torch.nn import functional as F
import networkx as nx

def normalize(A, symmetric=True):
    A = A + torch.eye(A.size(0))

    d = A.sum(1)

    if symmetric:
        D = torch.diag(torch.pow(d, -0.5))
        return D.mm(A).mm(D)
    else:
        D = torch.diag(torch.pow(d, -1))
        return D.mm(A)

class GCN(nn.Module):
    
    def __init__(self, A, input_dim, output_dim):
        super(GCN, self).__init__()
        self.A = A
        self.fc1 = nn.Linear(input_dim, input_dim, bias=False)
        self.fc2 = nn.Linear(input_dim, input_dim//2, bias=False)
        self.fc3 = nn.Linear(input_dim//2, output_dim, bias=False)
    
    def forward(self, X):
        X = F.relu(self.fc1(self.A.mm(X)))
        X = F.relu(self.fc2(self.A.mm(X)))
        return self.fc3(self.A.mm(X))
        
