
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient
import hashlib
import os

from config import MONGO_URI
client = MongoClient(MONGO_URI)

posts_col = client["content_management"]["posts"]
ml_db = client["ml_recommendation"]
ml_post_emb_col = ml_db.ml_post_embeddings

model = SentenceTransformer("all-MiniLM-L6-v2")

def hash_text(text):
    return hashlib.md5(text.encode("utf-8")).hexdigest()

def generate_post_embeddings():

    cursor = posts_col.find({}, {"_id": 1, "content": 1})

    for doc in cursor:
        post_id = doc["_id"]
        text = doc.get("content", "").strip()

        if not text:
            continue

        content_hash = hash_text(text)

        existing = ml_post_emb_col.find_one(
            {"postId": post_id},
            {"content_hash": 1}
        )

        if existing and existing.get("content_hash") == content_hash:
            continue

        emb = model.encode(text).tolist()

        ml_post_emb_col.update_one(
            {"postId": post_id},
            {"$set": {
                "postId": post_id,
                "embedding": emb,
                "content_hash": content_hash,
                "model": "all-MiniLM-L6-v2"
            }},
            upsert=True
        )
