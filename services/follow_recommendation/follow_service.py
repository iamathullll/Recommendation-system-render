import pickle
import os

from config import TOP_K_SERVE

FOLLOW_MODEL_PATH = "services/follow_recommendation/model/follow_recommendations.pkl"

precomputed_recs = None
popular_users = []
user_follow_map = {}


def load_follow_model():
    global precomputed_recs, popular_users

    if precomputed_recs is None:

        if not os.path.exists(FOLLOW_MODEL_PATH):
            print("⚠ Follow model not found. It will be created at startup.")
            return None

        print("Loading follow recommendation model...")

        with open(FOLLOW_MODEL_PATH, "rb") as f:
            precomputed_recs = pickle.load(f)

        popular_users = sorted(list(precomputed_recs.keys()))

        print("Follow model loaded successfully.")

    return precomputed_recs


def get_follow_recommendations(user_id, top_k=TOP_K_SERVE):

    model = load_follow_model()

    if model is None:
        return []

    user_id = str(user_id)
    already_following = user_follow_map.get(user_id, set())

    recs = model.get(user_id, [])

    if recs:
        return [
            u for u in recs
            if u != user_id and u not in already_following
        ][:top_k]

    return [
        u for u in popular_users
        if u != user_id and u not in already_following
    ][:top_k]