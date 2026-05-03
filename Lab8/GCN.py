import torch
import torch.nn as nn

# Dummy data (5 nodes, 10 features each)
x = torch.rand(5, 10)

# Adjacency matrix (connections)
adj = torch.tensor([
    [1,1,0,0,0],
    [1,1,1,0,0],
    [0,1,1,1,0],
    [0,0,1,1,1],
    [0,0,0,1,1]
], dtype=torch.float32)

class SimpleGCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x, adj):
        x = torch.matmul(adj, x)   # neighbor info
        x = self.fc(x)
        return x

model = SimpleGCN()
output = model(x, adj)

print("GCN Output:", output)
