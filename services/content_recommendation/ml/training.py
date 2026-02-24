
import os
import time
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import faiss
from pymongo import MongoClient
from torch.utils.data import Dataset, DataLoader

from recommendation_system.services.content_recommendation.ml.models import UserTower, ItemTower

# ==============================
# CONFIG
# ==============================
from config import MONGO_URI
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TEXT_EMB_DIM = 384
EMBED_DIM = 64
BPR_BATCH = 1024
NUM_EPOCHS = 5
BATCH_PROJ = 4096

BASE_DIR = os.path.dirname(__file__)  # services/content_recommendation/ml
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")

os.makedirs(ARTIFACT_DIR, exist_ok=True)


# ==============================
# DATA LOADING
# ==============================

def load_raw_embeddings():
    client = MongoClient(MONGO_URI)
    ml_db = client["ml_recommendation"]

    post_ids = []
    vectors = []

    cursor = ml_db["ml_post_embeddings"].find({}, {"postId": 1, "embedding": 1})

    for doc in cursor:
        post_ids.append(str(doc["postId"]))
        vectors.append(doc["embedding"])

    raw_vectors = np.array(vectors, dtype="float32")
    return post_ids, raw_vectors


def load_interactions(postid_to_idx):
    client = MongoClient(MONGO_URI)
    user_activity_col = client["user_management"]["user_activity"]

    users_set = set()
    pairs = []

    for doc in user_activity_col.find():
        uid = str(doc.get("userId"))
        users_set.add(uid)

        for field in ["likes", "comments", "shares", "savedPosts", "interested"]:
            for item in doc.get(field, []):
                pid = str(item.get("postId"))
                if pid in postid_to_idx:
                    pairs.append((uid, pid))

    users_list = sorted(users_set)
    user_to_idx = {u: i for i, u in enumerate(users_list)}

    idx_pairs = [
        (user_to_idx[u], postid_to_idx[p])
        for u, p in pairs
    ]

    return users_list, user_to_idx, idx_pairs


# ==============================
# DATASET
# ==============================

class RecDataset(Dataset):
    def __init__(self, pairs, num_items, user_positive_sets):
        self.pairs = pairs
        self.num_items = num_items
        self.user_positive_sets = user_positive_sets
        self.all_idxs = list(range(num_items))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        u, pos = self.pairs[idx]

        neg = random.choice(self.all_idxs)
        while neg in self.user_positive_sets[u]:
            neg = random.choice(self.all_idxs)

        return (
            torch.tensor(u, dtype=torch.long),
            pos,
            neg
        )


def bpr_loss(u, pos, neg):
    pos_scores = (u * pos).sum(dim=1)
    neg_scores = (u * neg).sum(dim=1)
    return -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()


# ==============================
# MAIN PIPELINE
# ==============================

def train_and_build():

    print("Loading raw embeddings...")
    post_ids, raw_vectors = load_raw_embeddings()

    postid_to_idx = {pid: i for i, pid in enumerate(post_ids)}

    print("Loading interactions...")
    users_list, user_to_idx, idx_pairs = load_interactions(postid_to_idx)

    num_users = len(users_list)
    num_items = len(post_ids)

    print("Users:", num_users, "Items:", num_items)

    # Build positive sets
    user_positive_sets = defaultdict(set)
    for u, p in idx_pairs:
        user_positive_sets[u].add(p)

    # Dataset
    dataset = RecDataset(idx_pairs, num_items, user_positive_sets)
    loader = DataLoader(dataset, batch_size=BPR_BATCH, shuffle=True)

    # Models
    user_tower = UserTower(num_users, EMBED_DIM).to(DEVICE)
    item_tower = ItemTower(TEXT_EMB_DIM, EMBED_DIM).to(DEVICE)

    optimizer = optim.Adam(
        list(user_tower.parameters()) +
        list(item_tower.parameters()),
        lr=1e-3
    )

    # ==========================
    # TRAIN
    # ==========================
    print("Training started...")
    user_tower.train()
    item_tower.train()

    for epoch in range(NUM_EPOCHS):
        total_loss = 0

        for user_idxs, pos_idxs, neg_idxs in loader:
            user_idxs = user_idxs.to(DEVICE)

            pos_vecs = torch.tensor(
                raw_vectors[pos_idxs.numpy()],
                dtype=torch.float32
            ).to(DEVICE)

            neg_vecs = torch.tensor(
                raw_vectors[neg_idxs.numpy()],
                dtype=torch.float32
            ).to(DEVICE)

            u_emb = user_tower(user_idxs)
            pos_emb = item_tower(pos_vecs)
            neg_emb = item_tower(neg_vecs)

            loss = bpr_loss(u_emb, pos_emb, neg_emb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} Loss: {total_loss:.4f}")

    print("Training completed.")

    # ==========================
    # SAVE MODELS
    # ==========================

    torch.save(user_tower.state_dict(),
               f"{ARTIFACT_DIR}/user_tower.pth")

    torch.save(item_tower.state_dict(),
               f"{ARTIFACT_DIR}/item_tower.pth")

    # ==========================
    # PROJECT ITEMS
    # ==========================

    print("Projecting item embeddings...")
    item_tower.eval()

    item_vectors = []

    with torch.no_grad():
        for i in range(0, len(raw_vectors), BATCH_PROJ):
            batch = torch.tensor(
                raw_vectors[i:i+BATCH_PROJ],
                dtype=torch.float32
            ).to(DEVICE)

            proj = item_tower(batch).cpu().numpy()
            item_vectors.append(proj)

    item_vectors = np.vstack(item_vectors).astype("float32")

    np.save(f"{ARTIFACT_DIR}/item_vectors.npy", item_vectors)
    np.save(f"{ARTIFACT_DIR}/post_ids.npy", np.array(post_ids))

    # ==========================
    # BUILD FAISS
    # ==========================

    print("Building FAISS index...")

    faiss.normalize_L2(item_vectors)

    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(item_vectors)

    faiss.write_index(
    index,
    os.path.join(ARTIFACT_DIR, "faiss.index"))

    print("Artifacts saved successfully.")
if __name__ == "__main__":
    train_and_build()
