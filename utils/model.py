import torch
import torch.nn as nn
from torch.utils.data import Dataset

import numpy as np

class NCF(nn.Module):
    def __init__(self, num_users, num_items):
        super().__init__()

        self.embedding_user_mlp = torch.nn.Embedding(
            num_embeddings=num_users, embedding_dim=8
        )
        self.embedding_item_mlp = torch.nn.Embedding(
            num_embeddings=num_items, embedding_dim=8
        )
        self.embedding_user_mf = torch.nn.Embedding(
            num_embeddings=num_users, embedding_dim=8
        )
        self.embedding_item_mf = torch.nn.Embedding(
            num_embeddings=num_items, embedding_dim=8
        )

        self.net = nn.Sequential(
            nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 8), nn.ReLU()
        )

        self.affine_output = torch.nn.Linear(in_features=8 + 8, out_features=1)
        self.logistic = torch.nn.Sigmoid()

    def forward(self, user_input, item_input):

        user_embedding_mlp = self.embedding_user_mlp(user_input)
        item_embedding_mlp = self.embedding_item_mlp(item_input)
        user_embedding_mf = self.embedding_user_mf(user_input)
        item_embedding_mf = self.embedding_item_mf(item_input)

        mlp_vector = torch.cat(
            [user_embedding_mlp, item_embedding_mlp], dim=-1
        )
        mf_vector = torch.mul(user_embedding_mf, item_embedding_mf)

        mlp_vector = self.net(mlp_vector)
        vector = torch.cat([mlp_vector, mf_vector], dim=-1)

        logits = self.affine_output(vector)
        rating = self.logistic(logits)

        return rating


class SongLensTrainDataset(Dataset):
    def __init__(self, ratings, all_songIds):
        self.users, self.items, self.labels = self.get_dataset(ratings, all_songIds)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.labels[idx]

    def get_dataset(self, ratings, all_movieIds):
        users, items, labels = [], [], []
        user_item_set = set(
            zip(ratings["session_id"], ratings["song_id"], ratings["listening_order"])
        )

        num_negatives = 4

        for u, i, _ in user_item_set:
            users.append(u)
            items.append(i)
            labels.append(1)
            for _ in range(num_negatives):
                negative_item = np.random.choice(all_songIds)
                while (u, negative_item) in user_item_set:
                    negative_item = np.random.choice(all_songIds)
                users.append(u)
                items.append(negative_item)
                labels.append(0)

        return torch.tensor(users), torch.tensor(items), torch.tensor(labels)
