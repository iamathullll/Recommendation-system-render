import os
import re
import gc
import pickle
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from services.collaborative_recommendation.data_loader import load_all_data
# ============================================================
# CONFIG
# ============================================================

DATA_PATH = "data/"
MODEL_PATH = "services/collaborative_recommendation/model/ppr_results.pkl"

POST_HALF_LIFE_DAYS = 30
SOCIAL_ALPHA = 0.7
TOP_POSTS = 100
POST_CHUNK = 512

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}")

# ============================================================
# LOAD DATA
# ============================================================

print("Loading data...")


users_df, posts_df, interactions_df = load_all_data()


# ============================================================
# CLEAN FOLLOWERS
# ============================================================

print("Cleaning followers...")

users_df = users_df.rename(columns={"_id": "user_id"})
users_df["user_id"] = users_df["user_id"].astype(str)

def clean_followers(x):
    if not isinstance(x, str) or x.strip() == "[]":
        return []
    user_ids = re.findall(r"ObjectId\('([a-f0-9]+)'\)", x)
    return [{"userId": uid} for uid in user_ids]

users_df["followers"] = users_df["followers"].apply(clean_followers)

# ============================================================
# CLEAN POSTS + TIME DECAY
# ============================================================

print("Processing posts...")

posts_df = posts_df.rename(columns={
    "_id": "post_id",
    "userId": "user_id",
    "createdAt": "created_at"
})

posts_df["post_id"] = posts_df["post_id"].astype(str)
posts_df["user_id"] = posts_df["user_id"].astype(str)

posts_df["created_at"] = pd.to_datetime(posts_df["created_at"], errors="coerce")
posts_df = posts_df.dropna(subset=["created_at"])

NOW = pd.Timestamp.now()

def time_decay(ts, half_life_days):
    delta_days = (NOW - ts).dt.total_seconds() / (3600 * 24)
    return np.exp(-np.log(2) * delta_days / half_life_days)

posts_df["decay_weight"] = time_decay(
    posts_df["created_at"],
    POST_HALF_LIFE_DAYS
)

# ============================================================
# CLEAN INTERACTIONS
# ============================================================

print("Processing interactions...")

interactions_df["user_id"] = interactions_df["user_id"].astype(str)
interactions_df["post_id"] = interactions_df["post_id"].astype(str)
interactions_df["interaction_score"] = interactions_df["interaction_score"].astype(float)

interactions_df = (
    interactions_df.groupby(["user_id", "post_id"], as_index=False)
    .interaction_score.sum()
)

# ============================================================
# INDEXING
# ============================================================

print("Building index maps...")

user_ids = users_df["user_id"].unique()
post_ids = posts_df["post_id"].unique()

uid_map = {u: i for i, u in enumerate(user_ids)}
pid_map = {p: i for i, p in enumerate(post_ids)}

U = len(user_ids)
P = len(post_ids)

print(f"Users: {U}")
print(f"Posts: {P}")

# ============================================================
# BUILD USER → POST GRAPH
# ============================================================

print("Building user-post graph...")

post_decay = dict(zip(posts_df["post_id"], posts_df["decay_weight"]))

ui = interactions_df["user_id"].map(uid_map)
pi = interactions_df["post_id"].map(pid_map)

interaction_weight = (
    interactions_df["interaction_score"] *
    interactions_df["post_id"].map(post_decay)
)

mask = (~ui.isna()) & (~pi.isna()) & (~interaction_weight.isna())

rows = ui[mask].astype(int).values
cols = pi[mask].astype(int).values
vals = interaction_weight[mask].astype(np.float32).values

# authorship edges
auth_ui = posts_df["user_id"].map(uid_map)
auth_pi = posts_df["post_id"].map(pid_map)
auth_w = posts_df["decay_weight"]

auth_mask = (~auth_ui.isna()) & (~auth_pi.isna())

rows = np.concatenate([rows, auth_ui[auth_mask].astype(int).values])
cols = np.concatenate([cols, auth_pi[auth_mask].astype(int).values])
vals = np.concatenate([vals, auth_w[auth_mask].astype(np.float32).values])

UP = sp.csr_matrix((vals, (rows, cols)), shape=(U, P), dtype=np.float32)

# normalize
row_sum = np.array(UP.sum(axis=1)).flatten()
inv = np.divide(1.0, row_sum, out=np.zeros_like(row_sum), where=row_sum != 0)
UP = UP.multiply(inv[:, None]).tocsr()

# ============================================================
# BUILD USER → USER GRAPH
# ============================================================

print("Building social graph...")

rows, cols = [], []

for _, r in users_df.iterrows():

    followed_user = r["user_id"]

    for f in r["followers"]:

        follower_id = f.get("userId")

        if follower_id in uid_map:

            rows.append(uid_map[follower_id])
            cols.append(uid_map[followed_user])

UU = sp.csr_matrix(
    (np.ones(len(rows), dtype=np.float32), (rows, cols)),
    shape=(U, U)
)

row_sum = np.array(UU.sum(axis=1)).flatten()
inv = np.divide(1.0, row_sum, out=np.zeros_like(row_sum), where=row_sum != 0)
UU = UU.multiply(inv[:, None]).tocsr()

# ============================================================
# CONVERT TO GPU SPARSE
# ============================================================

print("Moving graphs to GPU...")

def csr_to_gpu(csr):

    coo = csr.tocoo()

    indices = torch.from_numpy(
        np.vstack((coo.row, coo.col))
    ).long().to(DEVICE)

    values = torch.from_numpy(coo.data).float().to(DEVICE)

    return torch.sparse_coo_tensor(
        indices,
        values,
        csr.shape
    ).coalesce()

UP_gpu = csr_to_gpu(UP)
UP_T_gpu = UP_gpu.transpose(0, 1)
UU_gpu = csr_to_gpu(UU)

# ============================================================
# COMPUTE RECOMMENDATIONS
# ============================================================

print("Computing recommendations...")

results = {}

for start in range(0, P, POST_CHUNK):

    end = min(P, start + POST_CHUNK)

    print(f"Processing posts {start} → {end}")

    cols = torch.arange(start, end, device=DEVICE)

    E = torch.eye(P, device=DEVICE)[cols].T

    UP_chunk = torch.sparse.mm(UP_gpu, E)

    social_chunk = torch.sparse.mm(UU_gpu, UP_chunk)

    blended = SOCIAL_ALPHA * UP_chunk + (1 - SOCIAL_ALPHA) * social_chunk

    post_sim = torch.sparse.mm(UP_T_gpu, blended)

    scores = torch.sparse.mm(UP_gpu, post_sim).cpu().numpy()

    for u_idx, uid in enumerate(user_ids):

        top_idx = np.argsort(scores[u_idx])[::-1][:TOP_POSTS]

        results.setdefault(uid, []).extend(
            post_ids[start + i]
            for i in top_idx if start + i < P
        )

    # free memory
    del UP_chunk, social_chunk, blended, post_sim, scores
    torch.cuda.empty_cache()
    gc.collect()

# dedupe
for uid in results:
    results[uid] = list(dict.fromkeys(results[uid]))[:TOP_POSTS]

# ============================================================
# SAVE MODEL
# ============================================================

print("Saving model...")
with open(MODEL_PATH, "wb") as f:
    pickle.dump(results, f)

print("Model saved at:", MODEL_PATH)
print("TRAINING COMPLETE.")
