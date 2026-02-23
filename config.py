# ============================================
# Configuration File
# ============================================

import torch

import os
from dotenv import load_dotenv

load_dotenv()

# =========================
# MongoDB Config (GLOBAL)
# =========================

MONGO_URI = os.getenv("MONGO_URI")

USER_DB = "user_management"
USER_COLLECTION = "users"

POST_DB = "content_management"
POST_COLLECTION = "posts"

INTERACTION_DB = "user_personalization"
INTERACTION_COLLECTION = "userinteractions"


# ============================================
# Follow Recommendation Config
# ============================================

TOP_K_PRECOMPUTE = 100
TOP_K_SERVE = 100

W_SOCIAL = 0.5
W_SIM = 0.3
W_POP = 0.2


# ============================================
# Collaborative Recommendation Config
# ============================================

MODEL_PATH = "model/ppr_results.pkl"

POST_HALF_LIFE_DAYS = 30
SOCIAL_ALPHA = 0.7
POST_CHUNK = 512
TOP_POSTS = 100


# ============================================
# Device Config
# ============================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"