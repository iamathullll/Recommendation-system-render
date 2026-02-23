import os
from pymongo import MongoClient

from config import MONGO_URI
client = MongoClient(MONGO_URI)

posts_col = client["content_management"]["posts"]
user_activity_col = client["user_management"]["user_activity"]
ml_db = client["ml_recommendation"]
