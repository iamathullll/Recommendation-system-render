import torch.nn as nn

class UserTower(nn.Module):
    def __init__(self, num_users, embed_dim):
        super().__init__()
        self.emb = nn.Embedding(num_users, embed_dim)

    def forward(self, user_idxs):
        return self.emb(user_idxs)


class ItemTower(nn.Module):
    def __init__(self, in_dim, embed_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, embed_dim)

    def forward(self, x):
        return self.fc(x)
