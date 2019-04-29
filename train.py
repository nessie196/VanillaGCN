import networkx as nx
import torch
import pdb
import models
from torch.nn import functional as F

G = nx.karate_club_graph()
A = nx.adjacency_matrix(G).todense()

A_normed = models.normalize(torch.FloatTensor(A), True)

N = len(A)
X_dim = N

X = torch.eye(N, X_dim)
Y = torch.zeros(N, 1).long()

Y_mask = torch.zeros(N, 1, dtype=torch.uint8)

Y[0][0] = 0
Y[N-1][0] = 1

Y_mask[0][0] = 1
Y_mask[N-1][0] = 1

Real = torch.zeros(34, dtype=torch.long)

#pdb.set_trace()
for i in [1,2,3,4,5,6,7,8,11,12,13,14,17,18,20,22]:
    Real[i-1] = 0

for i in [9,10,15,16,19,21,23,24,25,26,27,28,29,30,31,32,33,34]:
    Real[i-1] = 1

gcn = models.GCN(A_normed, X_dim, 2)
gd = torch.optim.Adam(gcn.parameters())

for i in range(300):
    y_pred = F.softmax(gcn(X), dim=1)

    loss = (-y_pred.log().gather(1,Y.view(-1,1)))
    loss = loss.masked_select(Y_mask).mean()

    gd.zero_grad()
    loss.backward()
    gd.step()

    if i%20 == 0:
        _, mi = y_pred.max(1)
        print((mi == Real).float().mean().item())
