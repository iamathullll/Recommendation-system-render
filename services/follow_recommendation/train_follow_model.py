import pandas as pd
import numpy as np
import scipy.sparse as sp
import re
import pickle
from recommendation_system.services.collaborative_recommendation.data_loader import load_all_data
from config import *
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "follow_recommendations.pkl")
# ============================================================
# LOAD DATA
# ============================================================

print("Loading data...")


users_df, posts_df, interactions_df = load_all_data()


# =========================
# Clean Users
# =========================

users_df = users_df.rename(columns={"_id": "user_id"})
users_df["user_id"] = users_df["user_id"].astype(str)

def clean_followers(x):
    if not isinstance(x, str) or x.strip() == "[]":
        return []
    user_ids = re.findall(r"ObjectId\('([a-f0-9]+)'\)", x)
    return [{"userId": uid} for uid in user_ids]

users_df["followers"] = users_df["followers"].apply(clean_followers)
users_df = users_df[["user_id", "followers"]]

# =========================
# Clean Interactions
# =========================

interactions_df["user_id"] = interactions_df["user_id"].astype(str)
interactions_df["post_id"] = interactions_df["post_id"].astype(str)
interactions_df["interaction_score"] = interactions_df["interaction_score"].astype(float)

interactions_df = (
    interactions_df
    .groupby(["user_id", "post_id"], as_index=False)
    .interaction_score
    .sum()
)

# =========================
# Build Index
# =========================

user_ids = users_df["user_id"].unique()
uid_map = {u: i for i, u in enumerate(user_ids)}
U = len(user_ids)

# =========================
# Build UU Graph
# =========================

rows, cols = [], []

for _, r in users_df.iterrows():
    followed_user = r["user_id"]
    followers = r.get("followers", [])

    for f in followers:
        follower_id = f.get("userId")
        if follower_id in uid_map:
            rows.append(uid_map[follower_id])
            cols.append(uid_map[followed_user])

UU = sp.csr_matrix(
    (np.ones(len(rows)), (rows, cols)),
    shape=(U, U),
    dtype=np.float32
)

row_sum = np.array(UU.sum(axis=1)).flatten()
inv = np.divide(1.0, row_sum, out=np.zeros_like(row_sum), where=row_sum != 0)
UU = UU.multiply(inv[:, None]).tocsr()

# =========================
# Build UP Graph
# =========================

posts_df = posts_df.rename(columns={"_id": "post_id", "userId": "user_id"})
posts_df["post_id"] = posts_df["post_id"].astype(str)
posts_df["user_id"] = posts_df["user_id"].astype(str)

post_ids = posts_df["post_id"].unique()
pid_map = {p: i for i, p in enumerate(post_ids)}
P = len(post_ids)

ui = interactions_df["user_id"].map(uid_map)
pi = interactions_df["post_id"].map(pid_map)

mask = (~ui.isna()) & (~pi.isna())

rows = ui[mask].astype(int).values
cols = pi[mask].astype(int).values
vals = interactions_df.loc[mask, "interaction_score"].astype(np.float32).values

UP = sp.csr_matrix((vals, (rows, cols)), shape=(U, P), dtype=np.float32)

row_sum = np.array(UP.sum(axis=1)).flatten()
inv = np.divide(1.0, row_sum, out=np.zeros_like(row_sum), where=row_sum != 0)
UP = UP.multiply(inv[:, None]).tocsr()

# =========================
# Precompute Components
# =========================

user_popularity = np.array(UU.sum(axis=0)).flatten()
user_sim_matrix = UP.dot(UP.T)

# =========================
# Build Recommendations
# =========================

precomputed_recs = {}

for user_id in user_ids:

    u_idx = uid_map[user_id]
    direct = UU[u_idx].toarray().flatten()

    social_score = UU.dot(direct)
    social_score[u_idx] = 0
    social_score[direct > 0] = 0

    sim_score = user_sim_matrix[u_idx].toarray().flatten()
    sim_score[u_idx] = 0
    sim_score[direct > 0] = 0

    pop_score = user_popularity.copy()
    pop_score[u_idx] = 0
    pop_score[direct > 0] = 0

    def normalize(x):
        if x.max() > 0:
            return x / x.max()
        return x

    social_score = normalize(social_score)
    sim_score = normalize(sim_score)
    pop_score = normalize(pop_score)

    final_score = (
        W_SOCIAL * social_score +
        W_SIM * sim_score +
        W_POP * pop_score
    )

    top_users = np.argsort(final_score)[::-1][:TOP_K_PRECOMPUTE]

    precomputed_recs[user_id] = [
        user_ids[i]
        for i in top_users
        if final_score[i] > 0
    ]

# =========================
# Save Model
# =========================

with open(MODEL_PATH, "wb") as f:
    pickle.dump(precomputed_recs, f)

print("✅ Follow recommendations precomputed and saved.")
