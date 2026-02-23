import os
import time
import numpy as np
import faiss
from pymongo import MongoClient

MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)

ml_db = client["ml_recommendation"]
user_activity_col = client["user_management"]["user_activity"]

# ---------------------------
# LOAD DATA ON STARTUP
# ---------------------------

print("Loading projected item embeddings from Mongo...")

docs = list(ml_db["item_embeddings"].find({}, {"_id": 0}))

if not docs:
    raise ValueError("No item embeddings found in MongoDB.")

post_ids = [doc["postId"] for doc in docs]

item_vectors = np.array(
    [doc["item_embedding"] for doc in docs],
    dtype="float32"
)

print("Loaded", len(post_ids), "items.")

# Normalize once
faiss.normalize_L2(item_vectors)

# Build FAISS once
index = faiss.IndexFlatIP(item_vectors.shape[1])
index.add(item_vectors)

print("FAISS index ready. ntotal =", index.ntotal)

postid_to_idx = {pid: i for i, pid in enumerate(post_ids)}

# ---------------------------
# RECOMMEND FUNCTION
# ---------------------------

def recommend_for_user_full(user_id, top_k=100):

    start = time.time()

    doc = user_activity_col.find_one({"userId": user_id})
    if not doc:
        return {"error": "User not found"}

    liked_posts = [
        str(item.get("postId"))
        for item in doc.get("likes", [])
        if str(item.get("postId")) in postid_to_idx
    ]

    if not liked_posts:
        return {"error": "No interactions"}

    idxs = [postid_to_idx[pid] for pid in liked_posts]

    user_profile = item_vectors[idxs].mean(axis=0, keepdims=True)
    faiss.normalize_L2(user_profile)

    D, I = index.search(user_profile, top_k)

    results = [post_ids[i] for i in I[0]]

    return {
        "user_id": str(user_id),
        "final_posts": results,
        "time": time.time() - start
    }

