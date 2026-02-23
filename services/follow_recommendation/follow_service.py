import pickle
import numpy as np

from config import *

# Load precomputed recommendations
with open("services/follow_recommendation/model/follow_recommendations.pkl", "rb") as f:
    precomputed_recs = pickle.load(f)

# Build popularity fallback from precomputed data
all_users = list(precomputed_recs.keys())
popular_users = sorted(all_users)

# Example: Load user follow map if stored separately
user_follow_map = {}

def get_follow_recommendations(user_id, top_k=TOP_K_SERVE):

    user_id = str(user_id)
    already_following = user_follow_map.get(user_id, set())

    recs = precomputed_recs.get(user_id, [])

    if recs:
        return [
            u for u in recs
            if u != user_id and u not in already_following
        ][:top_k]

    return [
        u for u in popular_users
        if u != user_id and u not in already_following
    ][:top_k]
