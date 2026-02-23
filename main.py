import os
import subprocess
import sys

from fastapi import FastAPI
from pydantic import BaseModel

from services.reranking.rerank import rerank_posts
from services.follow_recommendation.follow_service import get_follow_recommendations


# ============================================================
# FASTAPI INIT
# ============================================================

app = FastAPI(title="Recommendation System API")


# ============================================================
# MODEL TRAINING UTIL
# ============================================================

def run_module(module_name, label):
    print(f"\n🚀 Training {label} model...\n")
    subprocess.run([sys.executable, "-m", module_name], check=True)
    print(f"✅ {label} training completed.\n")


def train_all_models():
    run_module("services.content_recommendation.ml.training", "Content")
    run_module("services.collaborative_recommendation.train", "Collaborative")
    run_module("services.follow_recommendation.train_follow_model", "Follow")


# ============================================================
# ENSURE MODELS EXIST
# ============================================================

def ensure_models_exist():

    required_files = [
        "services/content_recommendation/ml/artifacts/faiss.index",
        "services/collaborative_recommendation/model/ppr_results.pkl",
        "services/follow_recommendation/model/follow_recommendations.pkl",
    ]

    missing = [path for path in required_files if not os.path.exists(path)]

    if missing:
        print("⚠ Missing model artifacts detected:")
        for m in missing:
            print(f"   - {m}")

        print("\nRunning initial training...\n")
        train_all_models()
    else:
        print("✅ All model artifacts found. Skipping initial training.")


# ============================================================
# STARTUP EVENT
# ============================================================

@app.on_event("startup")
def startup_event():
    print("\n=== Recommendation System Starting ===\n")
    ensure_models_exist()
    print("🚀 System ready.\n")


# ============================================================
# REQUEST MODEL
# ============================================================

class UserRequest(BaseModel):
    user_id: str
    top_k: int = 100


# ============================================================
# RECOMMENDATION ENDPOINT
# ============================================================

@app.get("/recommend/{user_id}")
def recommend(user_id: str, top_k: int = 100):

    combined_posts = rerank_posts(user_id, top_k=top_k)
    follow_recs = get_follow_recommendations(user_id, top_k=50)

    return {
        "user_id": user_id,
        "post_recommendations": combined_posts,
        "follow_recommendations": follow_recs
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():
    return {"status": "ok"}