import os
import torch
# =====================================================
# MongoDB Config
# =====================================================

MONGO_URI = "mongodb+srv://athul:QmC0bZrYcMU6z4mP@magnifier-data.wfsxo9k.mongodb.net/?appName=magnifier-data"

USER_DB = "user_management"
USER_COLLECTION = "users"

POST_DB = "content_management"
POST_COLLECTION = "posts"

INTERACTION_DB = "user_personalization"
INTERACTION_COLLECTION = "userinteractions"


# =====================================================
# Model Config
# =====================================================

MODEL_PATH = "model/ppr_results.pkl"


# =====================================================
# Training Config
# =====================================================

POST_HALF_LIFE_DAYS = 30
SOCIAL_ALPHA = 0.7
POST_CHUNK = 512
TOP_POSTS = 100


# =====================================================
# Device Config
# =====================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
